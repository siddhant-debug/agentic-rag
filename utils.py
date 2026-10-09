from imports import Document, List


def format_docs(documents: List[Document]) -> str:
    """Join only the text of the documents for use in a prompt.

    Interpolating the raw list calls str() on every Document, which includes its
    metadata — for Unstructured chunks that is a huge base64 `orig_elements`
    blob (~155k chars for 4 chunks vs ~4k of real content). That slowed
    generation and buried the evidence the hallucination grader needs.
    """
    return "\n\n".join(d.page_content for d in documents)
