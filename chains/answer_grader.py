from imports import BaseModel, ChatNVIDIA, ChatPromptTemplate, Field, RunnableSequence, os


llm_client = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("NVIDIA_API"),
    temperature=0,
    timeout=60,
)


class AnswerGrade(BaseModel):
    """Binary score for correctness of answer"""

    binary_score: str = Field(description="Answer is correct or not, 'yes' or 'no' ")


structured_llm_grader = llm_client.with_structured_output(AnswerGrade).with_retry(
    stop_after_attempt=3, wait_exponential_jitter=True
)
system = """You are a grader assessing correctness of an answer to a user question. \n
    If the answer is correct and answers the user question, grade it as correct. \n
    If the answer is incorrect or does not answer the user question, grade it as incorrect. \n
    Give a binary score 'yes' or 'no' score to indicate whether the answer is correct or not."""

answer_grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question} \n\n Answer: {answer}"),
    ]
)

answer_grader: RunnableSequence = answer_grade_prompt | structured_llm_grader
