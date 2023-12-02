# This file consists of functions to be used for hosting the app
import pickle
print('In streamlit_app_functions file')
from langchain.embeddings import HuggingFaceInstructEmbeddings
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")

print('loading the retriever pickle file')
# Using the pickled retriver file
pickle_in = open("retriever_pycharm.pickle", "rb")
retriever = pickle.load(pickle_in)
pickle_in.close()
print('Pickle load is complete')
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