"""Single import page for the whole Agentic_RAG pipeline.

Every module under chains/, nodes/ and graph.py, ingestion.py and state.py gets its
stdlib and third-party names from here:

    from imports import ChatNVIDIA, ChatPromptTemplate, logging

Why one page: one place to see (and upgrade/replace) every external dependency,
and `load_dotenv()` runs exactly once instead of in every file.

Rules:
- Third-party + stdlib only. Project modules (state, consts, chains.*, nodes.*)
  are NOT re-exported here, to avoid circular imports.
- main.py and logging_config.py stay outside this page on purpose: they must run
  first and cheaply so the "started" log line appears before heavy imports load.
- Heavy ingestion-only libraries are loaded lazily (see _LAZY): `unstructured`,
  Chroma and the text splitters cost ~6 s to import, which would otherwise slow
  every offline chain test. They still live on this page; they just import on
  first use.
"""

# ── stdlib ────────────────────────────────────────────────────────────────
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Literal, TypedDict

# ── third-party (light, eager) ────────────────────────────────────────────
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_nvidia import ChatNVIDIA
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# One call for the whole project: every API key (NVIDIA_API, TAVILY_API_KEY,
# LANGSMITH_*) is available before any chain/node module builds its client.
load_dotenv()

# ── third-party (heavy, lazy: resolved on first `from imports import X`) ──
_LAZY = {
    "RecursiveCharacterTextSplitter": ("langchain_text_splitters", "RecursiveCharacterTextSplitter"),
    "UnstructuredLoader": ("langchain_unstructured", "UnstructuredLoader"),
    "Chroma": ("langchain_chroma", "Chroma"),
    "NVIDIAEmbeddings": ("langchain_nvidia_ai_endpoints", "NVIDIAEmbeddings"),
}


def __getattr__(name: str):
    """PEP 562 module-level hook: import heavy libs only when first requested."""
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        value = getattr(importlib.import_module(module), attr)
        globals()[name] = value  # cache so this hook runs once per name
        return value
    raise AttributeError(f"module 'imports' has no attribute {name!r}")


__all__ = [
    # stdlib
    "logging", "os", "Path", "Any", "Dict", "List", "Literal", "TypedDict",
    # third-party (eager)
    "load_dotenv", "Document", "StrOutputParser", "ChatPromptTemplate",
    "RunnableSequence", "ChatNVIDIA", "TavilySearch", "END", "StateGraph",
    "BaseModel", "Field",
    # third-party (lazy)
    *_LAZY,
]
