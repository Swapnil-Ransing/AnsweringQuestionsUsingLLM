# POC roject - EPFO Question and Answer System
End to End LLM POC project to answer the EPFO questions conssting of **Account (UAN) (creation, documents required, claims), KYC (procedure, update)**

This is an end to end LLM project based on Google Palm and Langchain. In this project a question and answer system related to EPFO (Employee's Provident Fund Organization) is developed. EPFO is one of the World's largest Social Security Organisations in terms of clientele and the volume of financial transactions undertaken. In the developed project questions related to account (UAN) (creation, documents required, claims), KYC (procedure, update) etc. are tried to answered using google palm large language model.

## Project Architecture:
1. CSV loading : CSV loader from langchain document loader will load the csv question and answer file.
2. Database questions embedding : Questions from CSV question and answer file will be embedded using huggingface embeding.
3. Vector Database : Embedded questions and corresponding answers will be stored using FAISS.
4. Creating a retrieval chain : Using a prompt template and google palm api retrieval chain will be prepared.

## Output:
Output will be an answer based on the input question. Following will happen in the background.
1. A question asked to the retrieval chain will try to find the similar questions from the vector database.
2. Corresponding answers from the vector database of the relevant questions from step 1 will be outputted nicely using google palm llm.
