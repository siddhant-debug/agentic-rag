from imports import Any, Dict, logging
from ingestion import retriever
from state import GraphState

logger = logging.getLogger(__name__)


def retrieve(state: GraphState) -> Dict[str, Any]:
    logger.info("NODE=retrieve | tool_called=chroma_retriever")
    question = state["question"]

    documnets = retriever.invoke(question)
    logger.info("NODE=retrieve | question=%r | docs_retrieved=%d", question, len(documnets))
    return {"documents": documnets, "question": question}
