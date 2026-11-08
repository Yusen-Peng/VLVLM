import statistics
import torch


class LLaVALatencyProfiler:
    """
    CUDA-event latency profiler for LLaVA.

    Prefill = first forward during generate().
    Decode = subsequent forwards.

    LLM prefill measures the LLaMA backbone only.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.enabled = False
        self.reset()

    def reset(self):
        self.forward_count = 0
        self.full_events = []
        self.llm_events = []
        self.full_starts = []
        self.llm_starts = []

    @staticmethod
    def _start():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @staticmethod
    def _end():
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _full_pre(self, module, args):
        if not self.enabled:
            return
        self.full_starts.append(self._start())

    def _full_post(self, module, args, output):
        if not self.enabled:
            return
        start = self.full_starts.pop()
        end = self._end()
        self.full_events.append(
            (self.forward_count, start, end)
        )
        self.forward_count += 1

    def _llm_pre(self, module, args):
        if not self.enabled:
            return
        self.llm_starts.append(self._start())

    def _llm_post(self, module, args, output):
        if not self.enabled:
            return
        start = self.llm_starts.pop()
        end = self._end()
        self.llm_events.append(
            (len(self.llm_events), start, end)
        )

    def install(self):
        self.handles.append(
            self.model.register_forward_pre_hook(self._full_pre)
        )
        self.handles.append(
            self.model.register_forward_hook(self._full_post)
        )

        # LLaVA: LlavaLlamaForCausalLM.model
        language_model = self.model.model

        self.handles.append(
            language_model.register_forward_pre_hook(self._llm_pre)
        )
        self.handles.append(
            language_model.register_forward_hook(self._llm_post)
        )

    def start(self):
        self.reset()
        self.enabled = True

    def stop(self):
        self.enabled = False
        torch.cuda.synchronize()

        full = [
            s.elapsed_time(e)
            for _, s, e in self.full_events
        ]

        llm = [
            s.elapsed_time(e)
            for _, s, e in self.llm_events
        ]

        return {
            "prefill_ms": full[0] if full else None,
            "llm_prefill_ms": llm[0] if llm else None,
            "decode_ms": sum(full[1:]),
            "decode_tpot_ms": (
                statistics.mean(full[1:])
                if len(full) > 1 else None
            ),
            "decode_steps": max(0, len(full) - 1),
            "total_forward_ms": sum(full),
        }

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()