
import torch
import torch.nn.functional as F
from typing import Optional, Union
import os
from transformers.cache_utils import Cache
from transformers.processing_utils import Unpack


from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLModel,
    Qwen3VLVisionModel,
    Qwen3VLModelOutputWithPast,
    TransformersKwargs,
    is_torchdynamo_compiling,
    Qwen3VLForConditionalGeneration,
    Qwen3VLCausalLMOutputWithPast
)


from .qwen3vl_compressed import (
    CompressedQwen3VLVisionPatchMerger,
    CompressedQwen3VLVisionModel,
    CompressedQwen3VLModel,
    CompressedQwen3VLForConditionalGeneration,
    CompressedQwen3VLModelOutput
)

SUPPORTED_STRATEGIES = {"DRIP", "Fixed"}

def get_image_lengths(grid_thw, merge_area=4):
    """Number of native patch tokens and merged visual tokensfor each image.

        grid_thw: [num_images, 3]

        Returns:
            patch_lengths: [num_images]
            mega_lengths:  [num_images]
    """
    patch_lengths = [int(v) for v in grid_thw.prod(dim=1).tolist()]
    if any(n % merge_area != 0 for n in patch_lengths):
        raise ValueError("Image patch counts must be divisible by merge_area.")

    mega_lengths = [n // merge_area for n in patch_lengths]
    return patch_lengths, mega_lengths


class BatchedQwen3VLVisionPatchMerger(CompressedQwen3VLVisionPatchMerger):
    """Batch-aware wrapper around the original merger.
    Each image is independently passed through the original compression implementation.
    The returned image features and boundaries are lists indexed by image.
    """
    def forward(
        self,
        x: torch.Tensor,
        grid_thw: torch.Tensor,
        inference: bool = False,
        **kwargs,
    ):
        if self.compressor is None:
            raise RuntimeError("Compressor must be initialized before inference.")

        strategy = self.compressor.merge_strategy

        if strategy not in SUPPORTED_STRATEGIES:
            raise NotImplementedError(f"Batch compression does not support {strategy}.")

        patch_lengths, mega_lengths = get_image_lengths(grid_thw, merge_area=self.merge_area)

        image_embeds = []
        image_boundaries = []
        boundary_losses = []
        offset = 0

        for image_idx, n_patches in enumerate(patch_lengths):
            image_hidden = x[offset:offset + n_patches]
            # Call the original merger implementation.
            compressed, boundaries, boundary_loss = super().forward(image_hidden, inference=inference)
            expected_tokens = mega_lengths[image_idx]

            if boundaries.shape != (1, expected_tokens):
                raise RuntimeError(f"Image {image_idx}: unexpected boundary shape {boundaries.shape}, expected (1, {expected_tokens}).")
            num_kept = int((boundaries[0] > 0.5).sum().item())

            if compressed.shape[0] != num_kept:
                raise RuntimeError(f"Image {image_idx}: {compressed.shape[0]} compressed features vs {num_kept} boundaries.")

            image_embeds.append(compressed)
            image_boundaries.append(boundaries)
            boundary_losses.append(boundary_loss)
            offset += n_patches

        if offset != x.shape[0]:
            raise RuntimeError(f"Image slicing mismatch: {offset} vs {x.shape[0]}")
        boundary_loss = torch.stack(boundary_losses).mean()
        return image_embeds, image_boundaries, boundary_loss


class BatchedQwen3VLVisionModel(CompressedQwen3VLVisionModel):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

        self.merger = BatchedQwen3VLVisionPatchMerger(config=config, use_postshuffle_norm=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        texts=None,
        **kwargs,
    ):
        hidden_states = self.patch_embed(hidden_states)

        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        seq_len, _ = hidden_states.size()

        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)

        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0,
            dtype=(grid_thw.dtype if torch.jit.is_tracing() else torch.int32))
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        # Vision blocks + DeepStack extraction
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if layer_num in self.deepstack_visual_indexes:
                idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = (self.deepstack_merger_list[idx](hidden_states))
                deepstack_feature_lists.append(deepstack_feature)

        # NEW: batch-aware image compression
        (
            image_embeds,
            boundaries,
            boundary_loss,
        ) = self.merger(
            hidden_states,
            grid_thw=grid_thw,
            inference=not self.training,
        )
        return image_embeds, deepstack_feature_lists, boundaries, boundary_loss





class BatchedQwen3VLModel(CompressedQwen3VLModel):
    def __init__(self, config):
        super().__init__(config)

        self.visual = BatchedQwen3VLVisionModel._from_config(config.vision_config)


    def get_image_features(self,pixel_values: torch.FloatTensor, image_grid_thw: Optional[torch.LongTensor] = None, texts=None):
        if image_grid_thw is None:
            raise ValueError("image_grid_thw is required.")
        pixel_values = pixel_values.to(dtype=self.visual.dtype)
        (
            image_embeds,
            deepstack_image_embeds,
            boundaries,
            boundary_loss,
        ) = self.visual(
            pixel_values,
            grid_thw=image_grid_thw,
            texts=texts,
        )
        # No single-image assertion!
        assert len(image_embeds) == image_grid_thw.shape[0]
        assert len(boundaries) == image_grid_thw.shape[0]
        return image_embeds, deepstack_image_embeds, boundaries, boundary_loss



    def compress_deepstack_features(self, deepstack_image_embeds, boundaries, image_grid_thw):
        """Apply each image's boundaries to its DeepStack features."""
        compressor = self.visual.merger.compressor
        if compressor is None:
            raise RuntimeError("Compressor is not initialized.")
        _, mega_lengths = get_image_lengths(image_grid_thw, merge_area=self.visual.merger.merge_area)

        if len(mega_lengths) != len(boundaries):
            raise ValueError("Number of images and boundary groups differ.")
        
        compressed_deepstack = []
        for deep_idx, deep in enumerate(deepstack_image_embeds):
            if deep.ndim != 2:
                raise ValueError(f"DeepStack {deep_idx} shape: {deep.shape}")
            compressed_parts = []
            offset = 0

            for image_idx, (n_mega, boundary) in enumerate(zip(mega_lengths, boundaries)):
                segment = deep[offset:offset + n_mega]

                if segment.shape[0] != n_mega:
                    raise RuntimeError(f"DeepStack {deep_idx}, image {image_idx}: invalid segment length.")
                compressed = compressor.apply_boundaries(segment.unsqueeze(0), boundary).squeeze(0)
                expected = int((boundary[0] > 0.5).sum().item())

                if compressed.shape[0] != expected:
                    raise RuntimeError(f"DeepStack {deep_idx}, image {image_idx}: token mismatch.")
                compressed_parts.append(compressed)
                offset += n_mega

            if offset != deep.shape[0]:
                raise RuntimeError(f"DeepStack {deep_idx}:consumed {offset}/{deep.shape[0]} tokens.")
            compressed_deepstack.append(torch.cat(compressed_parts, dim=0))
        return compressed_deepstack


    def build_sequence_keep_mask(self, image_mask, boundaries):
        """
        Convert per-image compression boundaries into a keep mask
        over the original batched LLM sequences.

        Args:
            image_mask:
                [B, L] boolean tensor marking image-token positions.

            boundaries:
                List of B tensors, each [1, N_i], where N_i is
                the original number of mega-tokens for image i.

        Returns:
            keep_mask:
                [B, L] boolean tensor.

                True  = retain this position.
                False = remove this position.
        """
        if image_mask.ndim != 2:
            raise RuntimeError(f"Expected image_mask [B, L], got {image_mask.shape}")

        B, L = image_mask.shape

        if len(boundaries) != B:
            raise ValueError(f"Expected {B} boundary tensors, got {len(boundaries)}")

        keep_mask = torch.ones_like(image_mask, dtype=torch.bool)

        for b in range(B):
            visual_positions = torch.where(image_mask[b])[0]
            boundary = boundaries[b][0] > 0.5
            if visual_positions.numel() != boundary.numel():
                raise RuntimeError(
                    f"Sample {b}: "
                    f"{visual_positions.numel()} visual placeholders "
                    f"vs {boundary.numel()} boundaries."
                )

            # Remove all original visual-token positions.
            keep_mask[b, visual_positions] = False

            # Restore positions selected by DRIP / Fixed.
            boundary_indices = torch.where(boundary)[0]
            kept_visual_positions = visual_positions[boundary_indices]
            keep_mask[b, kept_visual_positions] = True

        return keep_mask

    def pack_compressed_sequences(
        self,
        inputs_embeds,
        attention_mask,
        position_ids,
        image_mask,
        keep_mask,
    ):
        """Compact each sample independently and left-pad the compressed sequences into one batch."""
        B, L, D = inputs_embeds.shape

        if attention_mask is None:
            attention_mask = torch.ones(B, L, dtype=torch.long, device=inputs_embeds.device)

        if attention_mask.shape != (B, L):
            raise ValueError(f"Unexpected attention_mask shape: {attention_mask.shape}")
        if position_ids.shape != (3, B, L):
            raise ValueError(f"Unexpected position_ids shape: {position_ids.shape}")
        if image_mask.shape != (B, L):
            raise ValueError(f"Invalid image_mask shape: {image_mask.shape}")
        if keep_mask.shape != (B, L):
            raise ValueError(f"Invalid keep_mask shape: {keep_mask.shape}")

        # 1. Compact every sample independently
        compacted_embeds = []
        compacted_positions = []
        compacted_visual_masks = []

        for b in range(B):
            # Keep only actual tokens that survive compression.
            valid_keep = (keep_mask[b] & attention_mask[b].bool())
            indices = torch.where(valid_keep)[0]

            if indices.numel() == 0:
                raise RuntimeError(f"Sample {b} has no retained tokens.")
            
            compacted_embeds.append(inputs_embeds[b, indices])
            compacted_positions.append(position_ids[:, b, indices])
            compacted_visual_masks.append(image_mask[b, indices])

        # 2. Determine maximum compressed sequence length
        max_len = max(x.shape[0] for x in compacted_embeds)

        # 3. Allocate left-padded batch
        packed_embeds = inputs_embeds.new_zeros(B, max_len, D)
        packed_attention_mask = attention_mask.new_zeros(B, max_len)
        packed_position_ids = position_ids.new_zeros(3, B, max_len)

        packed_visual_mask = torch.zeros(B, max_len, dtype=torch.bool, device=inputs_embeds.device)

        # 4. Copy each compacted sequence to the right
        for b in range(B):
            n = compacted_embeds[b].shape[0]
            packed_embeds[b, -n:] = compacted_embeds[b]
            packed_attention_mask[b, -n:] = 1
            packed_position_ids[:, b, -n:] = compacted_positions[b]
            packed_visual_mask[b, -n:] = compacted_visual_masks[b]

        # 5. Sanity checks
        assert packed_embeds.shape == (B, max_len, D)
        assert packed_position_ids.shape == (3, B, max_len)
        assert torch.equal(packed_visual_mask.sum(dim=1), (image_mask & keep_mask).sum(dim=1))

        return packed_embeds, packed_attention_mask, packed_position_ids, packed_visual_mask



    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        texts=None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[tuple, CompressedQwen3VLModelOutput]:

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds."
            )

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        B, L, D = inputs_embeds.shape

        if pixel_values_videos is not None:
            raise NotImplementedError(
                "Batched compression currently supports images only."
            )

        # Initial implementation supports standard padded attention masks.
        if attention_mask is not None:
            if not isinstance(attention_mask, torch.Tensor) or attention_mask.ndim != 2:
                raise NotImplementedError(
                    "Expected a 2D attention_mask [B, L]."
                )
        boundary_loss = None
        keep_mask = None
        visual_pos_masks = None
        deepstack_visual_embeds = None
        # Image inputs are supplied only during multimodal prefill.
        is_image_prefill = pixel_values is not None


        # 2. Batched image encoding (MODIFIED)
        if is_image_prefill:
            if input_ids is None:
                raise NotImplementedError("Compressed image prefill requires input_ids.")

            if image_grid_thw is None:
                raise ValueError("image_grid_thw is required.")

            if image_grid_thw.shape[0] != B:
                raise NotImplementedError("Initial batched implementation requires exactly one image per sample.")
            (
                image_embeds,
                deepstack_image_embeds,
                boundaries,
                boundary_loss,
            ) = self.get_image_features(
                pixel_values,
                image_grid_thw,
                texts=texts,
            )

            # concatenate all compressed image features.
            compressed_image_embeds = torch.cat(image_embeds, dim=0)
            image_mask = input_ids == self.config.image_token_id

            if position_ids is not None:
                raise NotImplementedError("Initial compressed prefill expects position_ids=None so multimodal RoPE positions can be calculated here.")

            position_ids, original_rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                attention_mask=attention_mask,
            )

            # 4. Batched DeepStack compression (MODIFIED)
            compressed_deepstack = self.compress_deepstack_features(
                deepstack_image_embeds,
                boundaries,
                image_grid_thw,   # NEW required argument
            )

            # 5. Build per-sample keep masks (MODIFIED)
            keep_mask = self.build_sequence_keep_mask(
                image_mask,
                boundaries,
            )

            # 6. Compact and left-pad sequences (NEW)
            original_seq_len = L
            (
                inputs_embeds,
                attention_mask,
                position_ids,
                visual_pos_masks,
            ) = self.pack_compressed_sequences(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_mask=image_mask,
                keep_mask=keep_mask,
            )

            # Save the actual attention mask corresponding to the compressed KV cache. Reset it on every new image prefill.
            self._compressed_attention_mask = attention_mask.detach().clone()

            packed_seq_len = inputs_embeds.shape[1]

            # 7. Insert compressed visual embeddings (MOSTLY UNCHANGED)
            compressed_image_embeds = compressed_image_embeds.to(
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )

            num_visual_tokens = int(visual_pos_masks.sum().item())
            num_compressed_tokens = compressed_image_embeds.shape[0]
            num_boundary_tokens = sum(
                int((boundary[0] > 0.5).sum().item())
                for boundary in boundaries
            )

            if not (
                num_visual_tokens
                == num_compressed_tokens
                == num_boundary_tokens
            ):
                raise RuntimeError(
                    "POST-COMPRESSION mismatch:\n"
                    f"visual slots={num_visual_tokens}\n"
                    f"compressed features={num_compressed_tokens}\n"
                    f"boundary tokens={num_boundary_tokens}"
                )

            scatter_mask = visual_pos_masks.unsqueeze(-1).expand_as(
                inputs_embeds
            )

            inputs_embeds = inputs_embeds.masked_scatter(
                scatter_mask,
                compressed_image_embeds,
            )

            deepstack_visual_embeds = compressed_deepstack

            for i, deep in enumerate(deepstack_visual_embeds):
                if deep.shape[0] != num_visual_tokens:
                    raise RuntimeError(
                        f"DeepStack {i}: {deep.shape[0]} features "
                        f"vs {num_visual_tokens} visual positions."
                    )

            # 8. Physical KV-cache positions + multimodal RoPE (NEW)
            # The physical cache stores the padded compressed sequence, not the original uncompressed prompt.
            cache_position = torch.arange(packed_seq_len, device=inputs_embeds.device, dtype=torch.long)

            # get_rope_index computed the original deltas relative to
            # the ORIGINAL padded input length:
            #
            #   delta_original = logical_next_position - original_L
            #
            # After compression, the physical cache length changes.
            # Preserve the logical next position by updating:
            #
            #   delta_packed = logical_next_position - packed_L
            #
            self.rope_deltas = original_rope_deltas + (original_seq_len - packed_seq_len)

        else:
            # 9. Cached decoding (NEW)
            seq_length = inputs_embeds.shape[1]

            if past_key_values is not None:
                physical_past_len = past_key_values.get_seq_length()
            else:
                physical_past_len = 0

            # Use actual physical cache length rather than an uncompressed cache_position supplied by a caller.
            cache_position = torch.arange(
                physical_past_len,
                physical_past_len + seq_length,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

            # if attention_mask is not None:
            #     expected_mask_len = physical_past_len + seq_length
            #     if attention_mask.shape != (B, expected_mask_len):
            #         raise RuntimeError(
            #             "Attention mask is not aligned with the "
            #             "compressed KV cache. Expected shape "
            #             f"{(B, expected_mask_len)}, got "
            #             f"{tuple(attention_mask.shape)}. "
            #             "The generation wrapper must maintain "
            #             "the compressed attention mask."
            #         )


            if physical_past_len > 0:
                cached_mask = getattr(self, "_compressed_attention_mask", None)

                if cached_mask is None:
                    raise RuntimeError("Missing compressed attention mask. A compressed image prefill must run first.")

                if cached_mask.shape != (B, physical_past_len):
                    raise RuntimeError("Compressed attention mask/cache mismatch: "
                        f"mask={tuple(cached_mask.shape)}, "
                        f"cache_length={physical_past_len}"
                    )

                if attention_mask is None:
                    new_token_mask = torch.ones(B, seq_length, device=cached_mask.device, dtype=cached_mask.dtype)
                else:
                    if attention_mask.shape[0] != B:
                        raise RuntimeError("Generation batch size mismatch.")

                    new_token_mask = attention_mask[:, -seq_length:].to(device=cached_mask.device, dtype=cached_mask.dtype)

                # Preserve compressed prefill padding and append new tokens.
                attention_mask = torch.cat([cached_mask, new_token_mask], dim=1)
                self._compressed_attention_mask = attention_mask.detach().clone()
                expected_mask_len = physical_past_len + seq_length
                assert attention_mask.shape == (B, expected_mask_len), (
                    f"Expected {(B, expected_mask_len)}, "
                    f"got {tuple(attention_mask.shape)}"
                )

            if position_ids is None:
                if self.rope_deltas is None:
                    if input_ids is None:
                        raise ValueError("position_ids or input_ids is required.")
                    position_ids, self.rope_deltas = self.get_rope_index(
                        input_ids,
                        image_grid_thw,
                        video_grid_thw,
                        attention_mask=attention_mask,
                    )

                else:
                    deltas = self.rope_deltas.to(device=inputs_embeds.device, dtype=torch.long)
                    if deltas.shape != (B, 1):
                        raise RuntimeError(f"Unexpected rope_deltas shape: {deltas.shape}")

                    # [B, seq_length]
                    logical_positions = (cache_position.unsqueeze(0) + deltas)

                    # [3, B, seq_length]
                    position_ids = logical_positions.unsqueeze(0).expand(3, -1, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        
        return CompressedQwen3VLModelOutput(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            rope_deltas=self.rope_deltas,
            keep_mask=keep_mask,
            boundary_loss=boundary_loss,
        )





class BatchedQwen3VLForConditionalGeneration(CompressedQwen3VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.model = BatchedQwen3VLModel(config)

