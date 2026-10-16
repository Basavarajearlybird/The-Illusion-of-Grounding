"""Shared HuggingFace VLM backend for the Track A (white-box) model zoo.

LLaVA-1.6/NeXT and Qwen2/2.5-VL differ in prompt format and in the id used for
image placeholders, but they are identical in the two respects this project
depends on:

* both expand image placeholders into a contiguous span of *visual tokens* in
  ``input_ids``, so the span can be located generically by matching the model's
  own ``image_token_index``; and
* both run a decoder whose self-attention therefore attends over that span,
  which is what "cross-modal attention" means in these architectures and what
  Layer 3's Visual Attention Mass is computed over.

Implementing those once here is what keeps each concrete adapter to a ~40-line
subclass that only declares its prompt format.

Everything Track A needs is implemented on top of one primitive --
:meth:`_teacher_forced_logprobs` -- which returns per-token logprobs of an
arbitrary target string under an arbitrary context:

* Trust Index          -> score the visual answer and the textual answer.
* REFIND / CSR         -> score the same target with and without the docs.
* ECE / calibration    -> the mean logprob of the model's own generation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from gib.adapters.base import BaseAdapter, GenerationResult
from gib.eval.prompt import CAPTION_PROMPT, render_rag_prompt


class HFVLMAdapter(BaseAdapter):
    """Base class; subclasses set ``model_id`` and implement ``format_prompt``."""

    model_id: str = ""
    version: str = "1.0"
    capabilities = {"logprobs": True, "attention": True, "batch": False, "captioning": True}

    def __init__(
        self,
        model_id: str | None = None,
        *,
        device: str = "auto",
        dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        max_new_tokens: int = 32,
        attn_implementation: str = "eager",
    ):
        self.model_id = model_id or self.model_id
        self.name = self.name or self.model_id
        self.max_new_tokens = max_new_tokens
        self.load_in_4bit = load_in_4bit
        # `eager` attention is required for attention capture: the fused/SDPA
        # kernels do not materialise the attention matrix that Layer 3 hooks read.
        self.attn_implementation = attn_implementation
        self._device = device
        self._dtype = dtype
        self._model = None
        self._processor = None

    # -- lazy load ----------------------------------------------------------- #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[
            self._dtype
        ]
        kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "device_map": self._device,
            "attn_implementation": self.attn_implementation,
        }
        if self.load_in_4bit:
            # Quantisation measurably shifts hallucination rates, so it is only
            # ever a VRAM fallback and the runner stamps it into the results.
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_quant_type="nf4",
            )
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForVision2Seq.from_pretrained(self.model_id, **kwargs)
        self._model.eval()

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

    @property
    def processor(self):
        self._ensure_loaded()
        return self._processor

    # -- prompt formatting (subclass responsibility) -------------------------- #

    def format_prompt(self, text: str) -> str:
        """Wrap the canonical RAG prompt in the model's chat template."""
        raise NotImplementedError

    def _load_image(self, image_path: str):
        from PIL import Image

        return Image.open(image_path).convert("RGB")

    def _prepare(self, image_path: str | None, text: str):
        images = [self._load_image(image_path)] if image_path else None
        prompt = self.format_prompt(text) if image_path else text
        inputs = self.processor(images=images, text=prompt, return_tensors="pt")
        return {k: v.to(self.model.device) for k, v in inputs.items()}

    # -- visual token span ---------------------------------------------------- #

    def visual_token_span(self, input_ids) -> tuple[int, int]:
        """(start, end) of the visual-token span in ``input_ids``.

        Located by matching the model's own image-token id rather than by a
        hard-coded offset, so the same code is correct across LLaVA and Qwen and
        does not silently drift when a checkpoint changes its patch count.
        """

        cfg = self.model.config
        tok_id = getattr(cfg, "image_token_index", None)
        if tok_id is None:
            tok_id = getattr(cfg, "image_token_id", None)
        if tok_id is None:
            return (0, 0)
        mask = (input_ids[0] == tok_id).nonzero(as_tuple=True)[0]
        if mask.numel() == 0:
            return (0, 0)
        return (int(mask[0].item()), int(mask[-1].item()) + 1)

    # -- generation ------------------------------------------------------------ #

    def generate(
        self,
        image: str,
        question: str,
        docs: list[str],
        *,
        temperature: float = 0.0,
        capture_attention: bool = False,
        **kwargs: Any,
    ) -> GenerationResult:
        import torch

        t0 = time.time()
        text = render_rag_prompt(question, docs)
        inputs = self._prepare(image, text)
        prompt_len = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0,
            output_scores=True,
            return_dict_in_generate=True,
            output_attentions=capture_attention,
        )
        if temperature > 0:
            gen_kwargs.update(temperature=temperature, top_p=0.95)

        with torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out.sequences[0][prompt_len:]
        answer = self.processor.decode(new_tokens, skip_special_tokens=True).strip()

        # Per-token logprob of the token the model actually emitted -- this is
        # the confidence signal that ECE is computed against.
        logprobs: list[float] = []
        for step, score in enumerate(out.scores or []):
            if step >= new_tokens.shape[0]:
                break
            lp = torch.log_softmax(score[0].float(), dim=-1)
            logprobs.append(float(lp[new_tokens[step]].item()))

        attn = None
        if capture_attention and getattr(out, "attentions", None):
            attn = self._pack_attention(out, inputs)

        return GenerationResult(
            answer=answer,
            latency=time.time() - t0,
            token_logprobs=logprobs,
            attn_capture=attn,
            raw={"prompt_len": prompt_len, "n_new": int(new_tokens.shape[0])},
        )

    def _pack_attention(self, out, inputs) -> dict[str, np.ndarray]:
        """Reduce generate()'s attention tuples to per-layer/head visual mass.

        Full attention tensors are enormous (layers x heads x steps x seq). Only
        the mass over the visual span is ever needed downstream, so the reduction
        happens here rather than after a multi-GB round trip to disk.
        """
        import torch

        v0, v1 = self.visual_token_span(inputs["input_ids"])
        if v1 <= v0:
            return {}

        per_step = []
        for step_attn in out.attentions:  # one entry per generated token
            layers = torch.stack([a[0, :, -1, :].float() for a in step_attn])  # (L, H, seq)
            total = layers.sum(dim=-1).clamp_min(1e-9)
            visual = layers[:, :, v0:v1].sum(dim=-1)
            per_step.append((visual / total).cpu().numpy())  # (L, H)

        arr = np.stack(per_step)  # (steps, L, H)
        return {
            "vam_per_step_layer_head": arr.astype(np.float32),
            "visual_span": np.array([v0, v1], dtype=np.int32),
        }

    # -- Track A primitives ---------------------------------------------------- #

    def _teacher_forced_logprobs(
        self, image: str | None, context_text: str, target: str
    ) -> list[float]:
        """Per-token logprobs of ``target`` appended to ``context_text``.

        The target is tokenised *without* a leading special token and appended to
        the already-formatted prompt, so the scored positions correspond exactly
        to the answer tokens and not to any template scaffolding.
        """
        import torch

        inputs = self._prepare(image, context_text)
        prompt_ids = inputs["input_ids"]
        tgt_ids = self.processor.tokenizer(
            " " + target.strip(), return_tensors="pt", add_special_tokens=False
        ).input_ids.to(prompt_ids.device)

        full = torch.cat([prompt_ids, tgt_ids], dim=1)
        model_inputs = dict(inputs)
        model_inputs["input_ids"] = full
        if "attention_mask" in model_inputs:
            model_inputs["attention_mask"] = torch.ones_like(full)

        with torch.inference_mode():
            logits = self.model(**model_inputs).logits

        n_tgt = tgt_ids.shape[1]
        # Predicting token t uses the logits at position t-1.
        tgt_logits = logits[0, -n_tgt - 1 : -1, :].float()
        lp = torch.log_softmax(tgt_logits, dim=-1)
        return [float(lp[i, tgt_ids[0, i]].item()) for i in range(n_tgt)]

    def score_answer(
        self, image: str, question: str, docs: list[str], candidate: str, **kwargs: Any
    ) -> float:
        """Mean per-token logprob of ``candidate``.

        Length-normalised deliberately: the Trust Index compares a visual answer
        against a textual answer that may tokenise to different lengths ("apple"
        vs "orange"), and an unnormalised sum would confound modality preference
        with answer length.
        """
        lps = self._teacher_forced_logprobs(image, render_rag_prompt(question, docs), candidate)
        return float(np.mean(lps)) if lps else float("-inf")

    def token_logprobs_with_without_context(
        self, image: str, question: str, docs: list[str], target: str, **kwargs: Any
    ) -> tuple[list[float], list[float]]:
        with_ctx = self._teacher_forced_logprobs(
            image, render_rag_prompt(question, docs), target
        )
        without_ctx = self._teacher_forced_logprobs(
            image, render_rag_prompt(question, []), target
        )
        return with_ctx, without_ctx

    def caption(self, image: str, **kwargs: Any) -> str:
        import torch

        inputs = self._prepare(image, CAPTION_PROMPT)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=96, do_sample=False)
        new = out[0][inputs["input_ids"].shape[1] :]
        return self.processor.decode(new, skip_special_tokens=True).strip()

    def close(self) -> None:
        self._model = None
        self._processor = None
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:
            pass
