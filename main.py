import json
import logging
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from logging_config import setup_logging

log_path = setup_logging()
logger = logging.getLogger("main")
logger.info("STARTUP | main.py started, logging to %s", log_path)
logger.info("STARTUP | importing graph (loads Chroma + NVIDIA clients)")
_t0 = time.time()

from graph import app

logger.info("STARTUP | graph ready in %.1fs", time.time() - _t0)

DEFAULT_QUESTION = "What is agent memory?"  # used when you just press Enter
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
# Chunks carry a huge base64 `orig_elements` blob from the Unstructured loader;
# it's noise in the saved JSON, so it is dropped.
NOISY_METADATA = {"orig_elements"}


def to_jsonable(state: dict) -> dict:
    """Make the final graph state JSON-serialisable (Documents -> dicts)."""
    out = dict(state)
    out["documents"] = [
        {
            "page_content": d.page_content,
            "metadata": {
                k: v for k, v in d.metadata.items() if k not in NOISY_METADATA
            },
        }
        for d in state.get("documents", [])
    ]
    return out


def print_report(state: dict, elapsed: float) -> None:
    line = "=" * 72
    print(f"\n{line}\nQUESTION\n{line}\n{state['question']}")
    print(f"\n{line}\nANSWER\n{line}")
    print(state["generation"].strip())
    print(f"\n{line}\nSOURCES ({len(state['documents'])} docs used)\n{line}")
    for i, d in enumerate(state["documents"], 1):
        src = d.metadata.get("source") or "web search (Tavily)"
        print(f"{i}. {src}")
    print(f"\nweb_retries={state.get('retries', 0)} | elapsed={elapsed:.1f}s\n{line}")


if __name__ == "__main__":
    print("Hello Advanced RAG")
    # input() (not print) blocks and returns what you type; naming the variable
    # `question` also avoids shadowing the built-in `input`.
    question = input("Enter your question: ").strip() or DEFAULT_QUESTION
    logger.info("RUN | question=%r | invoking graph", question)
    t1 = time.time()  # start after typing so elapsed measures the graph only
    result = app.invoke(
        input={
            "question": question,
                "web_search": False,
            "retries": 0,
        },
        config={"recursion_limit": 10},
    )
    elapsed = time.time() - t1

    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"result_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(
        json.dumps(to_jsonable(result), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_report(result, elapsed)
    logger.info("RUN | raw JSON saved to %s", out_path)
    logger.info("RUN | terminal log saved to %s", log_path)
