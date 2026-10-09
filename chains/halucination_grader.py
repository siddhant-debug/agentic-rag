from imports import BaseModel, ChatNVIDIA, ChatPromptTemplate, Field, RunnableSequence, os

llm_client = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("NVIDIA_API"),
    temperature=0,
    timeout=60,
)

class GradeHallucination(BaseModel):
    """Binary score for hallucination of answer"""

    binary_score: str = Field(
        description="Answer is hallucinated or not, 'yes' or 'no' "
    )

structured_llm_grader = llm_client.with_structured_output(GradeHallucination).with_retry(
    stop_after_attempt=4, wait_exponential_jitter=True
)   

system = """You are a grader assessing hallucination of an answer to a user question. \n
    If the answer contains information that is not present in the retrieved document, grade it as hallucinated. \n
    Give a binary score 'yes' or 'no' score to indicate whether the answer is hallucinated or not."""

halcuniation_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question} \n\n Answer: {answer}"),
    ]
)       

halucination_grader: RunnableSequence = halcuniation_prompt | structured_llm_grader
