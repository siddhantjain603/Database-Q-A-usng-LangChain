from langchain_helper import get_few_shots_db_chain
import streamlit as st

st.title("AtliQ T Shirts: Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    chain = get_few_shots_db_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)