from dotenv import load_dotenv

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from langchain.llms import OpenAI

from utils import CFG, index_data, parse_source_docs, load_db


# CSV_PATH = os.path.join(CFG.DATA_DIR, CFG.CSV_FILE)
# doc_search = index_data(CSV_PATH)
doc_search = load_db()


def qa_chain(query):

    qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                     chain_type="stuff",
                                     return_source_documents=True,
                                     retriever=doc_search.as_retriever(),
                                     chain_type_kwargs={
        "prompt": PromptTemplate(
            template=CFG.CUSTOM_TEMPLATE,
            input_variables=["context", "question"],
        ),
    })

    result = qa({"query": query})

    ai_result = result['result']
    if CFG.REC_KEYWORD in ai_result:
        source_docs = result['source_documents']
        source_docs = parse_source_docs(source_docs)
    else:
        source_docs = ''

    return ai_result, source_docs


def conv_chain(query, chat_history):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=CFG.llm,
        retriever=doc_search.as_retriever(search_kwargs={'k': 4}),
        # condense_question_prompt=CONDENSE_TEMPLATE,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PromptTemplate(
            template=CFG.QA_TEMPLATE,
            input_variables=["context", "question"],
        ),
        }
    )

    result = qa_chain({"question": query, 'chat_history': chat_history})
    ai_result = result['answer']

    if CFG.REC_KEYWORD in ai_result:
        source_docs = result['source_documents']
        source_docs = parse_source_docs(source_docs)
    else:
        source_docs = ''

    return ai_result, source_docs
