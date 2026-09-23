import chromadb

from config.settings import get_embedding_model
from ingestion.chunker import Chunk

COLLECTION_NAME = "sba_corpus"


def get_chroma_collection(persist_dir: str = "chroma_db"):
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(COLLECTION_NAME)


def delete_chunks_for_source(source_doc: str, collection=None, persist_dir: str = "chroma_db") -> None:
    collection = collection if collection is not None else get_chroma_collection(persist_dir)
    collection.delete(where={"source_doc": source_doc})


def upsert_chunks(chunks: list[Chunk], collection=None, persist_dir: str = "chroma_db") -> int:
    if not chunks:
        return 0
    collection = collection if collection is not None else get_chroma_collection(persist_dir)
    embedder = get_embedding_model()
    embeddings = embedder.embed_documents([c.text for c in chunks])
    collection.upsert(
        ids=[c.chunk_id for c in chunks],
        embeddings=embeddings,
        documents=[c.text for c in chunks],
        metadatas=[
            {
                "source_doc": c.source_doc,
                "effective_date": c.effective_date,
                "program": c.program,
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ],
    )
    return len(chunks)
