# Pickle file generated from ipython notebook is not compatible with pycharm
# Pickle files are generated in pycharm for further use and hosting the streamlit app

# Importing the required packages
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings

# Initialize instructor embeddings using the Hugging Face model
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
print('Initialized the instructor embedding')
# for api key stored in .env file
from dotenv import load_dotenv
load_dotenv()
from langchain.llms import GooglePalm
import os
api_key = os.environ["GOOGLE_API_KEY"]
llm = GooglePalm(google_api_key=api_key, temperature=0.3)


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

print('Creating a promt template')
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


from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(llm=llm,
                            chain_type="stuff",
                            retriever=retriever,
                            input_key="query",
                            return_source_documents=True,
                            chain_type_kwargs=chain_type_kwargs)

print('Prompt template is created')
print('-'*50)

print('Check for an example')
print(chain("What if I purchase a mobile phone, do I need to create an account"))

print('-'*50)
print('Pickling the retriever file')
import pickle
# Lets save the retriever
pickle_out=open('retriever_pycharm.pickle', 'wb')
pickle.dump(retriever,pickle_out)
pickle_out.close()
print('Pickle is completed')
print('-'*50)