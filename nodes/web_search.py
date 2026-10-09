from imports import Any, Dict, Document, TavilySearch, logging
from state import GraphState

logger = logging.getLogger(__name__)

web_search_tool = TavilySearch(max_results=3)


def web_search(state: GraphState) -> Dict[str, Any]:
    logger.info("NODE=websearch | tool_called=tavily_search")
    question = state["question"]
    docs = list(state.get("documents") or [])
    retries = state.get("retries", 0) + 1
    tavilyresults = web_search_tool.invoke({"query": question})['results']
    logger.info("NODE=websearch | query=%r | results_returned=%d", question, len(tavilyresults))
    joined_tavily_result = "\n".join(
        [tavily_result["content"] for tavily_result in tavilyresults]
    )

    web_results = Document(page_content=joined_tavily_result)
    docs.append(web_results)

    logger.info("NODE=websearch | total_docs_after_merge=%d", len(docs))
    return {"documents": docs, "question": question, "retries": retries}
