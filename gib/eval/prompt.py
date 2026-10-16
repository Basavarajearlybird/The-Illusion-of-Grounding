"""Prompt rendering -- one canonical, frozen template.

Prompt-format sensitivity is one of the two silent failure modes the architecture
flags to guard against from the start. Every model in the zoo sees passages
framed identically, and the template version is logged with every result row, so
a format change can never be mistaken for a robustness finding.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from gib.config import CONFIG_DIR

PROMPT_VERSION = "rag_prompt_v1"


@lru_cache(maxsize=8)
def load_template(version: str = PROMPT_VERSION) -> str:
    path = Path(CONFIG_DIR) / "prompts" / f"{version}.txt"
    text = path.read_text(encoding="utf-8")
    # Strip the leading comment block; it documents the freeze, it is not prompt.
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    return "\n".join(lines).strip()


def render_context_block(docs: list[str]) -> str:
    """Number the passages so position is expressible in the model's input.

    Numbering is required for the position-bias ablation: `gt_doc_position` is
    logged in the dataset, and it is only a meaningful covariate if the position
    is actually visible to the model.
    """
    if not docs:
        return "(no passages were retrieved)"
    return "\n".join(f"[{i+1}] {d.strip()}" for i, d in enumerate(docs))


def render_rag_prompt(question: str, docs: list[str], version: str = PROMPT_VERSION) -> str:
    return load_template(version).format(
        context_block=render_context_block(docs), question=question.strip()
    )


CAPTION_PROMPT = (
    "Describe this image in two or three sentences. Mention every object you can "
    "identify."
)

POPE_TEMPLATE = "Is there {article} {obj} in the image? Answer yes or no."
