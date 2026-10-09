
from imports import Any, Dict, logging
from chains.genertation import generation_chain
from state import GraphState
from utils import format_docs

logger = logging.getLogger(__name__)


def generate(state: GraphState) -> Dict[str, Any]:
    logger.info("NODE=generate | tool_called=generation_llm")
    question = state["question"]
    documents = state["documents"]
    logger.info("NODE=generate | question=%r | context_docs=%d", question, len(documents))

    generation = generation_chain.invoke(
        {"context": format_docs(documents), "question": question}
    )
    logger.info("NODE=generate | generation_chars=%d", len(generation))
    return {"documents": documents, "question": question, "generation": generation}