import os
from utils import index_data
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate



DATA_DIR = "./Data"
CSV_FILE = "df_movies.csv"
CSV_PATH = os.path.join(DATA_DIR, CSV_FILE)



doc_search = index_data(CSV_PATH)

qa = RetrievalQA.from_chain_type(llm=OpenAI(),
                                 chain_type="stuff",
                                 return_source_documents=True,
                                 retriever=doc_search.as_retriever(),
                                 chain_type_kwargs={
                                                "prompt": PromptTemplate(
                                                  template=CUSTOM_TEMPLATE,
                                                  input_variables=["context", "question"],
                                                  ),
                                  })

def ask_chatbot():
    query = ""
    while query!= "QUIT":
        query = input("Chat here: ")
        if query == "QUIT":
            break
    # response = qa.run(query)
        result = qa({"query": query})

        print(f"chatbot: {result['result']}")
        print(f"sources: {result['source_documents']}")




if __name__ == "__main__":
    
    ask_chatbot()