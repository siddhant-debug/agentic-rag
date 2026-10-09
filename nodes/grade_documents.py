from imports import Any, Dict, logging
from chains.retirval_grader import retrieval_grader
from state import GraphState

logger = logging.getLogger(__name__)


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    logger.info("NODE=grade_documents | tool_called=retrieval_grader_llm")
    question = state["question"]
    documents = state["documents"]

    if not documents:
        logger.info("NODE=grade_documents | no documents retrieved -> flagging web_search")
        return {"question": question, "documents": [], "web_search": True}

    webSearch = False
    filteredDocs = []
    for i, docs in enumerate(documents):
        score = retrieval_grader.invoke(
            {"question": question, "document": docs.page_content}
        )

        grade = score.binary_score
        if grade.lower() == "yes":
            logger.info("NODE=grade_documents | doc=%d | relevant=yes", i)
            filteredDocs.append(docs)
        else:
            logger.info("NODE=grade_documents | doc=%d | relevant=no -> flagging web_search", i)
            webSearch = True
            continue

    logger.info(
        "NODE=grade_documents | kept=%d/%d | web_search=%s",
        len(filteredDocs), len(documents), webSearch,
    )
    return {"question": question, "documents": filteredDocs, "web_search": webSearch}
