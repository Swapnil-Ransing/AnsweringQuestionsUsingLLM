# This file consists of functions to be used for hosting the app
import pickle
print('In streamlit_app_functions file')
from langchain.embeddings import HuggingFaceInstructEmbeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
print('Initialized the instructor embedding')
# print('loading the retriever pickle file')
# # Using the pickled retriver file
# pickle_in = open("retriever_pycharm.pickle", "rb")
# retriever = pickle.load(pickle_in)
# pickle_in.close()
# print('Pickle load is complete')

# As pickle file is large, instead of using it, whole code is being used
# Importing the required packages
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load the data from EPFO faq's
from langchain.document_loaders.csv_loader import CSVLoader
print('-'*50)

print('Loading the data')
loader = CSVLoader(file_path='EPFO_FAQs.csv', encoding='unicode_escape', source_column="Question ")

# Store the loaded data in the 'data' variable
data = loader.load()

# correcting the rows as there are only specific number of questions
data=data[:41]
print('Data load is complete')
print('-'*50)

# Create a FAISS instance for vector database from 'data'
vectordb = FAISS.from_documents(documents=data,embedding=instructor_embeddings)
print('Vector database is prepared')
# Create a retriever for querying the vector database
retriever = vectordb.as_retriever(score_threshold = 0.7)
print('Retriever is created')

print('-'*50)
# adding the details about googl epalm and api key
from dotenv import load_dotenv
load_dotenv()

from langchain.llms import GooglePalm
import os
api_key = os.environ["GOOGLE_API_KEY"]

llm = GooglePalm(google_api_key=api_key, temperature=0.3)

print('Creating a prompt template')
# creating the prompt template
from langchain.prompts import PromptTemplate

prompt_template = """Given the following context and a question, generate an answer based on this context only.
In the answer try to provide as much text as possible from "Answer" section in the source document context without making much changes.
If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

CONTEXT: {context}

QUESTION: {question}"""


PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)
chain_type_kwargs = {"prompt": PROMPT}
print('Prompt template is completed')
print('-'*50)

from langchain.chains import RetrievalQA

# creating a function to be used for retrieving the responses
def get_qa_chain():
    chain_r = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs=chain_type_kwargs)
    return (chain_r)