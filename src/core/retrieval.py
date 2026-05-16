# pyrefly: ignore [missing-import]
from rank_bm25 import BM25Okapi
from src.config.set import settings
from src.core.vector_store import load_vectorstore
from src.core.preprocess import preprocess_documents

vectorstore = load_vectorstore()

documents = preprocess_documents(
    settings.RAW_DATA_PATH
)

bm25_corpus = [
    doc.page_content.lower().split()
    for doc in documents
]

bm25 = BM25Okapi(bm25_corpus)

def vector_search(
    query: str,
    k: int = settings.TOP_K_RESULTS
):

    results = vectorstore.similarity_search(
        query,
        k=k
    )

    return results


def keyword_search(
    query: str,
    k: int = settings.TOP_K_RESULTS
):

    tokenized_query = query.lower().split()

    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    results = [
        documents[i]
        for i in ranked_indices
    ]

    return results


def hybrid_search(
    query: str,
    k: int = settings.TOP_K_RESULTS
):

    vector_results = vector_search(query, k=k)

    keyword_results = keyword_search(query, k=k)


    combined_results = {}


    # ADD VECTOR RESULTS
    for doc in vector_results:

        doc_id = doc.metadata["entity_id"]

        combined_results[doc_id] = doc


    # ADD BM25 RESULTS
    for doc in keyword_results:

        doc_id = doc.metadata["entity_id"]

        if doc_id not in combined_results:

            combined_results[doc_id] = doc


    final_results = list(
        combined_results.values()
    )


    return final_results[:k]