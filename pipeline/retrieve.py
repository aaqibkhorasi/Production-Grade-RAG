from config.settings import get_embedding_model
from ingestion.embed import get_chroma_collection
from pipeline.state import Citation, QueryState

TOP_K = 5


def retrieve(state: QueryState, collection=None) -> QueryState:
    collection = collection if collection is not None else get_chroma_collection()
    embedder = get_embedding_model()
    query_embedding = embedder.embed_query(state["question"])
    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    retrieved: list[Citation] = [
        Citation(
            source_doc=metadata["source_doc"],
            effective_date=metadata["effective_date"],
            program=metadata["program"],
            chunk_text=text,
        )
        for text, metadata in zip(documents, metadatas)
    ]
    return {**state, "retrieved_chunks": retrieved}
