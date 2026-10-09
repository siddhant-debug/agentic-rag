from imports import ChatNVIDIA, ChatPromptTemplate, StrOutputParser, os

llm = ChatNVIDIA(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("NVIDIA_API"),
    temperature=0,
    timeout=60,
    # NVIDIA's free-tier NIM catalog queues requests under load, so a single
    # call can hang far past a "normal" timeout — retry with backoff instead
    # of failing the whole chain on one slow response.
).with_retry(stop_after_attempt=4, wait_exponential_jitter=True)

# Inline equivalent of hub.pull("rlm/rag-prompt") — the hub module was removed
# from the langchain package, and pulling public prompts now requires a
# LangSmith API key.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n"
            "Question: {question} \nContext: {context} \nAnswer:",
        )
    ]
)

generation_chain = prompt | llm | StrOutputParser()