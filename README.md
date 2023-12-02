# CHATBOT POC Project - EPFO Question and Answer System [CHATBOT LINK](https://chatbotllmepfo.streamlit.app/)
End to End LLM POC project to answer the EPFO questions conssting of **Account (UAN) (creation, documents required, claims), KYC (procedure, update)**

This is an end to end LLM project based on **Google Palm and Langchain**. In this project a question and answer system related to EPFO (Employee's Provident Fund Organization) is developed. EPFO is one of the World's largest Social Security Organisations in terms of clientele and the volume of financial transactions undertaken. In the developed project questions related to account (UAN) (creation, documents required, claims), KYC (procedure, update) etc. are tried to answered using google palm large language model.

## Project Architecture:
1. CSV loading : CSV loader from **langchain document loader** will load the csv question and answer file.
2. Database questions embedding : Questions from CSV question and answer file will be embedded using **huggingface embeding**.
3. Vector Database : Embedded questions and corresponding answers will be stored using **FAISS**.
4. Creating a retrieval chain : Using a **prompt template and google palm** api retrieval chain will be prepared.

## Output:
Output will be an answer based on the input question. Following will happen in the background.
1. A question asked to the retrieval chain will try to find the similar questions from the vector database.
2. Corresponding answers from the vector database of the relevant questions from step 1 will be outputted nicely using google palm llm.

## Discussion with examples:
### 1. Example 1
**Question :** What is the procedure to change the password ? 
For this question, similar questions found out by retriever are:
  a. In which format I should create my UAN password
  b. What to do if I forgot my password
  c. What to do if I forgot my password and my registered mobile with UAN has also changed
  d. Can I change my already seeded Bank account number
Each of the answers are assessed by llm and appropriate answer with nice formatting as follows is provided:
**Answer :**   Please click on “Change Password” at Member Interface of Unified Portal. Provide your UAN with CAPTCHA. System will send the OTP on your mobile which is seeded with UAN and you can reset the password.

Following is an app image for this question:
