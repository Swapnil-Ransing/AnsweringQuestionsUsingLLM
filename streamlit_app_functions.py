# This file consists of functions to be used for hosting the app
import streamlit as st
print('In streamlit_app_functions file')

import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY=st.secrets["google_api_key"]
# LANGCHAIN_API_KEY=st.secrets["langchain_api_key"]
# os.environ["LANGCHAIN_TRACING_V2"]="true"
# LANGCHAIN_PROJECT=st.secrets["langchain_project"]
HF_TOKEN=st.secrets["huggingface_access_token"]

print('Loaded the api keys')


# Importing the required packages
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load the data from EPFO faq's
from langchain_community.document_loaders import CSVLoader
print('-'*50)

print('Loading the data')
loader = CSVLoader('EPFO_FAQs.csv', encoding='unicode_escape', source_column="Question ")

# Store the loaded data in the 'data' variable
data = loader.load()

# correcting the rows as there are only specific number of questions
data=data[:41]
print('Data load is complete')
print('-'*50)

from langchain_huggingface import HuggingFaceEmbeddings
print("Imported HuggingFaceEmbeddings package")

embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
print('Initialized the embedding')

# Create a FAISS instance for vector database from 'data'
vectordb = FAISS.from_documents(documents=data,embedding=embeddings)
print('Vector database is prepared')
# Create a retriever for querying the vector database
retriever = vectordb.as_retriever()
print('Retriever is created')

print('-'*50)

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-001",
    temperature=0.3,
    max_retries=2
)

print('Creating a prompt template')
# creating the prompt template

system_prompt = (
    """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "Answer" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT : {context} """
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)

print('Prompt template is completed')
print('-'*50)

# creating a function to be used for retrieving the responses
def get_qa_chain():
    chain_r = create_retrieval_chain(retriever, question_answer_chain)
    return (chain_r)
