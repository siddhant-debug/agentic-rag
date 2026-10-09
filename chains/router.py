from imports import BaseModel, ChatNVIDIA, ChatPromptTemplate, Field, Literal, os

class RouterInput(BaseModel):
    """Router user query to most relevant data source"""
    datasource: Literal["vectorstore", "websearch"] = Field(
        description="The data source to use for answering the user query. "
        "If the user query is a question that can be answered with retrieved documents, "
        "use 'vectorstore'. If the user query is a question that requires web search, "
        "use 'websearch'."
    )  


llm_client = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("NVIDIA_API"),
    temperature=0,
    timeout=60,
)

structured_llm_router = llm_client.with_structured_output(RouterInput).with_retry(
    stop_after_attempt=4, wait_exponential_jitter=True
)

system = """You are an expert for routing user queries to the most relevant data source.
You have access to two data sources: a vectorstore and a web search engine.
If the user query is a question that can be answered with retrieved documents, use 'vectorstore'.
If the user query is a question that requires web search, use 'websearch'.
You must respond with a JSON object that contains the following fields:
- datasource: a string that is either 'vectorstore' or 'websearch', indicating the data source to use for answering the user query.
"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User query: {query}"),
    ]
)

query_router = router_prompt | structured_llm_router


