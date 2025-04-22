# This file is the app python file to be run from the terminal
import streamlit as st
from streamlit_app_functions import get_qa_chain

# setting up a title and description for the app
st.title("EPFO Question Portal")
st.text("Questions related to following can be answered through this portal:")
st.text("1. Account (UAN) (creation, documents required, claims)")
st.text("2. KYC (procedure, update)")

question = st.text_input("Question: ")

if st.button('Submit'):
    chain = get_qa_chain()
    response = chain.invoke({"input":question})

    st.header("Answer")
    st.write(response['answer'])

st.markdown('##') # Adding the vetical seperating space
st.markdown('##') # Adding the vetical seperating space
st.markdown('##') # Adding the vetical seperating space

st.subheader("Explore The Details:")
st.write("For git repository and details, [check out this link](https://github.com/Swapnil-Ransing/AnsweringQuestionsUsingLLM)")
st.write("Created by [Swapnil Ransing](https://swapnil-ransing.github.io/PersonalWebsite/)"
         " using Gemini gemini-2.0-flash-001 ChatGoogleGenerativeAI , Huggingface embedding model all-mpnet-base-v2, FAISS and langchain")
st.write("For other projects from Swapnil, [check out this link](https://swapnil-ransing.github.io/PersonalWebsite/projects/)")
