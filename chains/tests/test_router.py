"""Unit tests for Agentic_RAG/chains/router.py — no live network calls.

Same approach as the other chain tests: the structured-output step is
monkeypatched, so NVIDIA_API is never needed. The router has two outcomes,
'vectorstore' and 'websearch'; every behavioural test is parametrized over
both so each scenario is validated.
"""

import pytest
from pydantic import ValidationError

import router
from router import RouterInput

DATASOURCES = ["vectorstore", "websearch"]


# --- schema (RouterInput) ------------------------------------------------


@pytest.mark.parametrize("datasource", DATASOURCES)
def test_router_input_accepts_both_valid_datasources(datasource):
    assert RouterInput(datasource=datasource).datasource == datasource


def test_router_input_requires_datasource():
    with pytest.raises(ValidationError):
        RouterInput()


@pytest.mark.parametrize("bad", ["web", "VectorStore", "", "database"])
def test_router_input_rejects_unknown_datasource(bad):
    # Literal[...] means the LLM can never route somewhere the graph can't handle.
    with pytest.raises(ValidationError):
        RouterInput(datasource=bad)


# --- prompt --------------------------------------------------------------


def test_router_prompt_declares_only_query_placeholder():
    assert set(router.router_prompt.input_variables) == {"query"}


def test_router_prompt_formats_query():
    formatted = router.router_prompt.format(query="unique-query-marker")
    assert "unique-query-marker" in formatted


def test_router_system_prompt_names_both_datasources():
    system_text = router.router_prompt.messages[0].prompt.template
    assert "vectorstore" in system_text
    assert "websearch" in system_text


def test_router_chain_is_runnable():
    assert hasattr(router.query_router, "invoke")


# --- chain behaviour (LLM mocked) ----------------------------------------


@pytest.mark.parametrize("datasource", DATASOURCES)
def test_query_router_invoke_returns_structured_datasource(monkeypatch, datasource):
    def fake_invoke(_self, _input, config=None, **kwargs):
        return RouterInput(datasource=datasource)

    # structured_llm_router is a RunnableRetry (pydantic model) — patch the
    # class, not the instance. monkeypatch restores it after the test.
    monkeypatch.setattr(type(router.structured_llm_router), "invoke", fake_invoke)

    result = router.query_router.invoke({"query": "any question"})

    assert isinstance(result, RouterInput)
    assert result.datasource == datasource


def test_query_router_routes_each_query_to_its_own_datasource(monkeypatch):
    """Both scenarios through ONE chain: the fake LLM decides from the prompt
    text it receives, proving the query actually reaches the model."""

    def fake_invoke(_self, prompt_value, config=None, **kwargs):
        text = prompt_value.to_string()
        if "latest news" in text:
            return RouterInput(datasource="websearch")
        return RouterInput(datasource="vectorstore")

    monkeypatch.setattr(type(router.structured_llm_router), "invoke", fake_invoke)

    in_domain = router.query_router.invoke({"query": "What is agent memory?"})
    out_of_domain = router.query_router.invoke({"query": "latest news on RAG"})

    assert in_domain.datasource == "vectorstore"
    assert out_of_domain.datasource == "websearch"


@pytest.mark.parametrize("datasource", DATASOURCES)
def test_query_router_passes_formatted_prompt_to_llm(monkeypatch, datasource):
    captured = {}

    def fake_invoke(_self, prompt_value, config=None, **kwargs):
        captured["text"] = prompt_value.to_string()
        return RouterInput(datasource=datasource)

    monkeypatch.setattr(type(router.structured_llm_router), "invoke", fake_invoke)

    router.query_router.invoke({"query": "unique-query-marker"})

    assert "unique-query-marker" in captured["text"]
    assert "routing user queries" in captured["text"]


def test_query_router_invalid_llm_output_is_not_silently_accepted():
    # An out-of-range value can't even be constructed, so a bad LLM answer
    # fails loudly at the schema instead of mis-routing the graph.
    with pytest.raises(ValidationError):
        RouterInput(datasource="somewhere_else")
