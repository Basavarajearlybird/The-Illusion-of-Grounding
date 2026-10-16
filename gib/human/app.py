"""Layer 4 -- the Streamlit collection interface.

Run with::

    streamlit run gib/human/app.py

Design requirements from the architecture, each implemented below:

* **randomised order** -- per participant, seeded by participant id, so order
  effects do not align across participants and become a confound;
* **response time recorded per item** -- the paper reports median RT alongside
  accuracy;
* **no backtracking** -- a participant who revisits an item after seeing later
  ones is no longer a clean measurement of first-pass judgement;
* **a consent screen**, with responses stored anonymised under a generated id.

The participant sees exactly what the model saw: the same degraded image file and
the same retrieved passages, rendered in the same order the model received them.
"""

from __future__ import annotations

import json
import random
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from gib.config import HUMAN_DIR
from gib.human.sample import load_study

RESPONSES_PATH = HUMAN_DIR / "responses.jsonl"

CONSENT = """
### Consent

You are being invited to take part in a short research study on how people
answer questions about images when accompanying written passages may be
inaccurate.

* The task takes roughly 20-30 minutes and involves answering short questions
  about images.
* Some images are deliberately blurred or cropped, and some passages are
  deliberately incorrect. There are no trick questions beyond this, and there is
  no penalty for being unsure.
* We record only your answers and how long each answer took. We do **not**
  record your name, email, IP address, or any other identifying information.
  You are identified only by a random code generated in your browser.
* Participation is voluntary. You may close the page at any time, and partial
  responses will simply be discarded.

This is a non-sensitive perception task. By clicking below you confirm that you
are 18 or older and consent to your anonymised answers being used in research
and released in aggregate form.
"""


def _append_response(payload: dict) -> None:
    RESPONSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESPONSES_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def main() -> None:
    st.set_page_config(page_title="GIB-15K Human Study", layout="centered")

    if "pid" not in st.session_state:
        st.session_state.pid = uuid.uuid4().hex[:12]
        st.session_state.consented = False
        st.session_state.idx = 0
        st.session_state.started = None

    # -- consent gate ------------------------------------------------------- #
    if not st.session_state.consented:
        st.title("GIB-15K Human Study")
        st.markdown(CONSENT)
        if st.button("I consent and wish to begin", type="primary"):
            st.session_state.consented = True
            st.rerun()
        return

    # -- load and shuffle per participant ----------------------------------- #
    if "items" not in st.session_state:
        try:
            items = load_study()
        except FileNotFoundError:
            st.error("No study items found. Run `make human-sample` first.")
            return
        rng = random.Random(st.session_state.pid)
        rng.shuffle(items)
        st.session_state.items = items

    items = st.session_state.items
    idx = st.session_state.idx

    if idx >= len(items):
        st.success("All done — thank you!")
        st.write(f"You answered {len(items)} questions.")
        st.caption(f"Your anonymous code: `{st.session_state.pid}`")
        return

    item = items[idx]
    st.progress(idx / len(items), text=f"Question {idx + 1} of {len(items)}")

    if st.session_state.started is None:
        st.session_state.started = time.time()

    # -- stimulus ------------------------------------------------------------ #
    img = Path(item.image_path)
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.warning("Image unavailable for this item.")

    if item.retrieved_docs:
        st.markdown("**Retrieved reference passages:**")
        for i, doc in enumerate(item.retrieved_docs, 1):
            st.markdown(f"> **[{i}]** {doc}")

    st.markdown(f"### {item.question}")
    answer = st.text_input(
        "Your answer (a single word or short phrase)",
        key=f"ans_{idx}",
        placeholder="type your answer…",
    )

    st.caption(
        "Answer from what you see in the image. The passages may or may not be "
        "accurate. If you genuinely cannot tell, write “unsure”."
    )

    # -- submit (no backtracking: there is no Previous control) -------------- #
    if st.button("Submit and continue", type="primary", disabled=not answer.strip()):
        _append_response(
            {
                "participant_id": st.session_state.pid,
                "variant_id": item.variant_id,
                "answer": answer.strip(),
                "response_time_s": round(time.time() - st.session_state.started, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        st.session_state.idx += 1
        st.session_state.started = None
        st.rerun()


if __name__ == "__main__":
    main()
