# pyrefly: ignore [missing-import]
from langchain_chroma import Chroma

from src.config.set import settings
from src.config.embeddings import get_embedding_model


COLLECTION_NAME = "shl_assessments"


def create_vectorstore(documents):

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(

        documents=documents,

        embedding=embedding_model,

        collection_name=COLLECTION_NAME,

        persist_directory=settings.CHROMA_DB_DIR
    )

    return vectorstore


def load_vectorstore():

    embedding_model = get_embedding_model()

    vectorstore = Chroma(

        collection_name=COLLECTION_NAME,

        embedding_function=embedding_model,

        persist_directory=settings.CHROMA_DB_DIR
    )

    return vectorstore