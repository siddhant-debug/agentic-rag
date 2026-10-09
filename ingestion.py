from imports import (
    Chroma,
    NVIDIAEmbeddings,
    Path,
    RecursiveCharacterTextSplitter,
    UnstructuredLoader,
    os,
)

# Anchor to this file, not the CWD: a relative "./.chroma" opens whatever
# .chroma sits in the directory you ran from (the repo root has an old,
# iCloud-evicted one), which fails with "pool timed out".
CHROMA_DIR = str(Path(__file__).resolve().parent / ".chroma")

embeddings = NVIDIAEmbeddings(
    model="nvidia/nemotron-3-embed-1b", api_key=os.getenv("NVIDIA_API")
)

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]


vectorstore = Chroma(
    collection_name="agenticrag",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR,
)

# Guard the scrape/embed step: it only needs to run once. Without this check,
# re-importing ingestion.py (i.e. every program run, since retrieve.py imports
# `retriever` at module load) would re-scrape all 3 URLs and re-embed them via
# the NVIDIA API on every single run, piling up duplicate vectors each time.
if vectorstore._collection.count() == 0:
    docs = [
        UnstructuredLoader(
            web_url=url, chunking_strategy="basic", max_characters=1000000
        ).load()
        for url in urls
    ]
    docs_list = [item for sublist in docs for item in sublist]

    textsplitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=250, chunk_overlap=0
    )

    docsplits = textsplitter.split_documents(docs_list)

    vectorstore = Chroma.from_documents(
        documents=docsplits,
        collection_name="agenticrag",
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )

retriever = vectorstore.as_retriever()
