from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_groq import ChatGroq

load_dotenv(dotenv_path="a.env")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY") #uncomment this and use
GROQ_API_KEY=st.secrets['GROQ_API_KEY'] #comment this , iam using this for hoisiting the app



# Guard API key
if not GROQ_API_KEY:
    st.warning("Missing GROQ_API_KEY. Please add it to a.env")
    st.stop()

# LLM + Embeddings
llm = ChatGroq(
    model="Gemma2-9b-It",
    groq_api_key=GROQ_API_KEY,
    temperature=0.2,
    max_tokens=512,
)

def rag_chain(llm):
    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Rewrite the latest question as standalone using chat history. Do not answer."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, st.session_state.retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([

        ("system",
        "The user has uploaded a PDF. Use ONLY the retrieved context from that PDF to answer the question.\n\n"
        "Answer using the retrieved context. If unknown, say only 'i don't know'. \n\ncontext:{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    core_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return core_chain,'answer'


def chain(llm):
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Chat normally, considering prior messages."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    core_chain = LLMChain(llm=llm, prompt=chat_prompt)
    return core_chain,'text'

