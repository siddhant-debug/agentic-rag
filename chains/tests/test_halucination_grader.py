"""Unit tests for Agentic_RAG/chains/halucination_grader.py — no live network calls.

Same approach as test_chains.py: the structured-output step is monkeypatched,
so NVIDIA_API is never needed. Covers import/wiring, the Pydantic schema,
prompt formatting, and that the chain passes the grader's output through.
"""

import pytest
from pydantic import ValidationError

import halucination_grader
from halucination_grader import GradeHallucination


def test_grade_hallucination_requires_binary_score():
    with pytest.raises(ValidationError):
        GradeHallucination()


def test_grade_hallucination_accepts_binary_score():
    assert GradeHallucination(binary_score="no").binary_score == "no"


def test_halucination_prompt_declares_document_question_answer_placeholders():
    assert set(halucination_grader.halcuniation_prompt.input_variables) == {
        "document",
        "question",
        "answer",
    }


def test_halucination_prompt_formats_all_three_inputs():
    formatted = halucination_grader.halcuniation_prompt.format(
        document="unique-document-marker",
        question="unique-question-marker",
        answer="unique-answer-marker",
    )
    assert "unique-document-marker" in formatted
    assert "unique-question-marker" in formatted
    assert "unique-answer-marker" in formatted


def test_halucination_prompt_system_message_defines_yes_no_grading():
    system_text = halucination_grader.halcuniation_prompt.messages[0].prompt.template
    assert "hallucinat" in system_text
    assert "'yes' or 'no'" in system_text


def test_halucination_grader_chain_is_runnable():
    assert hasattr(halucination_grader.halucination_grader, "invoke")


@pytest.mark.parametrize("score", ["yes", "no"])
def test_halucination_grader_invoke_returns_structured_score(monkeypatch, score):
    def fake_invoke(_self, _input, config=None, **kwargs):
        return GradeHallucination(binary_score=score)

    # structured_llm_grader is a RunnableRetry (pydantic model) — patch the
    # class, not the instance.
    monkeypatch.setattr(
        type(halucination_grader.structured_llm_grader), "invoke", fake_invoke
    )

    result = halucination_grader.halucination_grader.invoke(
        {"document": "doc text", "question": "a question", "answer": "an answer"}
    )

    assert isinstance(result, GradeHallucination)
    assert result.binary_score == score


def test_halucination_grader_passes_formatted_prompt_to_llm(monkeypatch):
    captured = {}

    def fake_invoke(_self, prompt_value, config=None, **kwargs):
        captured["text"] = prompt_value.to_string()
        return GradeHallucination(binary_score="no")

    monkeypatch.setattr(
        type(halucination_grader.structured_llm_grader), "invoke", fake_invoke
    )

    halucination_grader.halucination_grader.invoke(
        {
            "document": "unique-document-marker",
            "question": "unique-question-marker",
            "answer": "unique-answer-marker",
        }
    )

    for marker in ("document", "question", "answer"):
        assert f"unique-{marker}-marker" in captured["text"]
