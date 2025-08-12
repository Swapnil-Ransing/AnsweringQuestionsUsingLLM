# This file consists of functions to be used for hosting the app
print('In streamlit_app_functions file')
import streamlit as st
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT")
os.environ["LANGSMITH_TRACING"] = "true"

print('Loaded the api keys')

# Importing the required packages
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chat_models import init_chat_model

# Load the data from EPFO faq's embedding
db_path = "faiss_db_epfo"
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = FAISS.load_local(db_path, embeddings, allow_dangerous_deserialization=True)

print("FAISS DB loaded from:", os.path.abspath(db_path))
print('-'*50)

# Create a retriever for querying the vector database
retriever = vectordb.as_retriever()
print('Retriever is created')

print('-'*50)

llm = init_chat_model("openai:gpt-4.1-nano", temperature=0.3,max_retries=2)

print('Creating a prompt template')
# creating the prompt template

prompt_template = """Given the following context and a question, generate an answer based on this context only. Context is 
relevent FAQ's found from employee provident fund FAQ's database. Read the context in details and come up with the answer from 
the "Answer" section in the context source document. You can elaborate your answers in maximum 2 sentences. If the relevent context
is not found to the asked question, kindly state "I do not know.".

CONTEXT: {context}

QUESTION: {question}"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}


chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)

print('Prompt template is completed')
print('-'*50)
