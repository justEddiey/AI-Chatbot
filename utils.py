from typing import List
import json

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.llms import OpenAI
from langchain.chat_models import ChatVertexAI, ChatOpenAI


class CFG:
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613", max_tokens=500)
    CUSTOM_TEMPLATE = """
            You are a helpful recommendation chatbot, recommending best movies based on description by user.
            if question doesnt contain preference or specification on what movies, ask the user for more details.
            Using the following context recommend movies to the user with detailed desccriptionse created from the movie attributes
            {context}

            If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to .
            {question}

            """
    QA_TEMPLATE = """You are a helpful recommendation chatbot, recommending best gifts based on description by user.
                    Always politely ask the user for more descriptions about the person he wants to send the gift to, if no description is given.
                    If {question} doesn't provide a budget ask the user for their allocated budget.
                    Using the following context to recommend a list of products to the user with short descriptions, starting with the words "Here are my suggestions".
                    If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
                    {context}

                    If the question is not related to recommending or buying gifts, products or appliances, politely respond that you are tuned to only answer questions that are related to .
                    {question}
                    Helpful answer in markdown:"""

    DATA_DIR = "./Data"
    CSV_FILE = "df_movies.csv"
    DB_DIR = "db"
    REC_KEYWORD = 'suggestions'


def csv_to_text(file_path: str) -> List[str]:
    """Splits Csv File contents into Chunks of text for indexing"""
    loader = CSVLoader(file_path=file_path, encoding="iso-8859-1")
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

# use only first 200 rows of document to save cost in creating embeddings
    texts = texts[:200]
    return texts


def json_to_text(file_path: str) -> List[str]:
    loader = JSONLoader(
        file_path=file_path,
        jq_schema=".[]",
        text_content=False)
# loader = CSVLoader(file_path=file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # use only first 200 rows of document to save cost in creating embeddings
    texts = texts[:200]


def index_data(texts) -> Chroma:
    """Indexes data in  Vector Store."""

    embeddings = OpenAIEmbeddings()
    # docsearch = Chroma.from_documents(texts, embeddings)
    docsearch = FAISS.from_documents(texts, embeddings)

    return docsearch


def load_db():
    """loads a persisted vector database using openAI embeddings"""
    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.load_local(CFG.DB_DIR, embeddings)

    return docsearch


def parse_source_docs(source_doc):
    """Transforms source docs fron retrieval chain to list of dictionaries"""
    source_docs = []
    for doc in source_doc:
        source_dict = json.loads(doc.page_content)
        source_docs.append(source_dict)
    return source_docs
