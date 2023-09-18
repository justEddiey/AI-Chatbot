from typing import List

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Chroma


def csv_to_text(file_path: str) -> List[str]:
    """Splits Csv File contents into Chunks of text for indexing"""
    loader = CSVLoader(file_path=file_path, encoding="iso-8859-1")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

# use only first 200 rows of document to save cost in creating embeddings
    texts = texts[:200]
    return texts


def index_data(file_path: str) -> Chroma:
    """Indexes data in  Vector Store."""
    texts = csv_to_text(file_path)

    embeddings = OpenAIEmbeddings()
    # docsearch = Chroma.from_documents(texts, embeddings)
    docsearch = FAISS.from_documents(texts, embeddings)

    return docsearch


