
from imports import END, StateGraph, logging
from chains.halucination_grader import halucination_grader
from chains.router import query_router
from consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from nodes import generate, grade_documents, retrieve, web_search
from state import GraphState
from utils import format_docs
from chains.answer_grader import answer_grader



MAX_WEB_RETRIES = 1

logger = logging.getLogger(__name__)


def decide_to_generate(state):
    logger.info("ROUTER=decide_to_generate | web_search_flag=%s", state["web_search"])

    if state["web_search"]:
        logger.info("ROUTER=decide_to_generate | route=%s (irrelevant docs found)", WEBSEARCH)
        return WEBSEARCH
    else:
        logger.info("ROUTER=decide_to_generate | route=%s (all docs relevant)", GENERATE)
        return GENERATE


def grade_generated_in_docsandques(state: GraphState):
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]
    retries = state.get("retries", 0)

    # hallucination grader: "yes" means the answer IS hallucinated. It always
    # runs, even at the retry cap, so every answer gets a logged faithfulness
    # verdict (before, capped runs skipped grading entirely).
    hallucinated = (
        halucination_grader.invoke(
            {
                "document": format_docs(documents),
                "question": question,
                "answer": generation,
            }
        ).binary_score.strip().lower()
        == "yes"
    )
    logger.info("ROUTER=grade_generated | hallucinated=%s", hallucinated)

    # Hard cap: each retry costs 3+ slow NVIDIA calls, so without this a
    # grader that never says "useful" loops generate <-> websearch forever.
    if retries >= MAX_WEB_RETRIES:
        logger.info(
            "ROUTER=grade_generated | retries=%d >= cap=%d -> END (hallucinated=%s, accepted at cap)",
            retries, MAX_WEB_RETRIES, hallucinated,
        )
        return "usefull"

    if hallucinated:
        return "not_supported"

    useful = (
        answer_grader.invoke(
            {"question": question, "answer": generation}
        ).binary_score.strip().lower()
        == "yes"
    )
    logger.info("ROUTER=grade_generated | useful=%s", useful)
    return "usefull" if useful else "not_usefull"


def route_query_to_datasource(state: GraphState):
    """Entry router: decide whether to start from the vectorstore or the web."""
    query = state["question"]
    logger.info("ROUTER=route_query_to_datasource | tool_called=router_llm | query=%r", query)
    try:
        datasource = query_router.invoke({"query": query}).datasource
    except Exception as exc:
        # The router is only an optimisation; if its LLM call fails (timeout,
        # 5xx after retries) the safe default is the local vectorstore.
        logger.warning(
            "ROUTER=route_query_to_datasource | router failed (%s: %s) -> fallback=%s",
            type(exc).__name__, exc, RETRIEVE,
        )
        return RETRIEVE

    if datasource == "websearch":
        logger.info("ROUTER=route_query_to_datasource | datasource=%s -> route=%s", datasource, WEBSEARCH)
        return WEBSEARCH

    if datasource != "vectorstore":
        logger.warning(
            "ROUTER=route_query_to_datasource | unexpected datasource=%r -> fallback=%s",
            datasource, RETRIEVE,
        )
    else:
        logger.info("ROUTER=route_query_to_datasource | datasource=%s -> route=%s", datasource, RETRIEVE)
    return RETRIEVE


workflow = StateGraph(GraphState)
workflow.add_node(RETRIEVE,retrieve)
workflow.add_node(GRADE_DOCUMENTS,grade_documents)
workflow.add_node(GENERATE,generate)
workflow.add_node(WEBSEARCH,web_search)

workflow.set_conditional_entry_point(
    route_query_to_datasource,
    path_map={
        RETRIEVE:RETRIEVE,
        WEBSEARCH:WEBSEARCH
    }   
)
workflow.add_edge(RETRIEVE,GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    path_map={
        WEBSEARCH:WEBSEARCH,
        GENERATE:GENERATE
    }
)
workflow.add_conditional_edges(
    GENERATE,
    grade_generated_in_docsandques,
    path_map={
        'usefull':END,
        'not_usefull':WEBSEARCH,
        'not_supported':WEBSEARCH
    }
)
workflow.add_edge(WEBSEARCH,GENERATE)


app = workflow.compile()
logger.info("GRAPH | compiled | entry=router | max_web_retries=%d", MAX_WEB_RETRIES)

try:
    # mermaid.ink is a network call; never let it block or break an import.
    app.get_graph().draw_mermaid_png(output_file_path="graph.png")
except Exception as exc:
    logger.warning("graph.png not rendered: %s", exc)

