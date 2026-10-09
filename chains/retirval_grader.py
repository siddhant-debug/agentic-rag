from imports import BaseModel, ChatNVIDIA, ChatPromptTemplate, Field, os

llm_client = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("NVIDIA_API"),
    temperature=0,
    timeout=60,
)


class GradeDocument(BaseModel):
    """Binary score for relevance of document"""

    binary_score: str = Field(
        description="Document is relevant to the question , 'yes' or 'no' "
    )


# with_structured_output only exists on the raw chat model, so it must be
# called before with_retry (which wraps the result in a generic Runnable
# that no longer exposes with_structured_output). NVIDIA's free-tier NIM
# catalog queues requests under load, so retry with backoff instead of
# failing the whole chain on one slow response.
structured_llm_grader = llm_client.with_structured_output(GradeDocument).with_retry(
    stop_after_attempt=4, wait_exponential_jitter=True
)
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)


retrieval_grader = grade_prompt | structured_llm_grader
