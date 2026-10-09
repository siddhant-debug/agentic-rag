"""Unit tests for Agentic_RAG/chains — no live network calls.

Every ChatNVIDIA/structured-output call is monkeypatched, so these tests
never need NVIDIA_API set and run offline. They cover two things per chain:
1. The chain imports and wires together (this is what actually caught the
   `binary_score` missing-type-annotation bug in retirval_grader.py — that
   error only surfaces at import time, not at call time).
2. Prompt formatting and output-parsing behave as expected, independent of
   what the LLM itself would return.
"""

from langchain_core.messages import AIMessage
from pydantic import ValidationError
import pytest

import genertation
import retirval_grader
from retirval_grader import GradeDocument


# --- generation chain (chains/genertation.py) ---------------------------


def test_generation_chain_imports_and_is_runnable():
    assert hasattr(genertation.generation_chain, "invoke")


def test_generation_prompt_declares_question_and_context_placeholders():
    assert set(genertation.prompt.input_variables) == {"question", "context"}


def test_generation_chain_invoke_returns_plain_string_from_llm_output(monkeypatch):
    def fake_invoke(_self, _input, config=None, **kwargs):
        return AIMessage(content="mocked answer")

    # ChatNVIDIA is a pydantic model, so its instances don't allow setting
    # arbitrary attributes that aren't declared fields — patch the bound
    # method on the class instead of the instance.
    monkeypatch.setattr(type(genertation.llm), "invoke", fake_invoke)

    result = genertation.generation_chain.invoke(
        {"question": "What is LangGraph?", "context": "LangGraph is a graph library."}
    )

    # StrOutputParser should unwrap the AIMessage down to its .content string.
    assert result == "mocked answer"


def test_generation_chain_passes_question_and_context_into_prompt(monkeypatch):
    captured = {}

    def fake_invoke(_self, prompt_value, config=None, **kwargs):
        # prompt_value is a ChatPromptValue; convert to text to inspect it.
        captured["text"] = prompt_value.to_string()
        return AIMessage(content="ok")

    monkeypatch.setattr(type(genertation.llm), "invoke", fake_invoke)

    genertation.generation_chain.invoke(
        {"question": "unique-question-marker", "context": "unique-context-marker"}
    )

    assert "unique-question-marker" in captured["text"]
    assert "unique-context-marker" in captured["text"]


# --- retrieval grader chain (chains/retirval_grader.py) ------------------


def test_grade_document_requires_binary_score():
    with pytest.raises(ValidationError):
        GradeDocument()


def test_grade_document_accepts_binary_score():
    doc = GradeDocument(binary_score="yes")
    assert doc.binary_score == "yes"


def test_grade_prompt_formats_document_and_question():
    formatted = retirval_grader.grade_prompt.format(
        document="unique-document-marker", question="unique-question-marker"
    )
    assert "unique-document-marker" in formatted
    assert "unique-question-marker" in formatted


def test_retrieval_grader_invoke_returns_structured_score(monkeypatch):
    def fake_invoke(_self, _input, config=None, **kwargs):
        return GradeDocument(binary_score="yes")

    # structured_llm_grader is a RunnableRetry (pydantic model, like
    # ChatNVIDIA above) — patch the class, not the instance.
    monkeypatch.setattr(
        type(retirval_grader.structured_llm_grader), "invoke", fake_invoke
    )

    result = retirval_grader.retrieval_grader.invoke(
        {"document": "some retrieved text", "question": "some question"}
    )

    assert isinstance(result, GradeDocument)
    assert result.binary_score == "yes"
