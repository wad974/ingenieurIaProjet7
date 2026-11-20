import streamlit as st
from chatbot import chatbot

st.set_page_config(page_title="Événements - La Réunion", page_icon="🎭")
st.title("Assistant événements - La Réunion")
st.caption("Concert, expo, atelier… posez votre question.")

question = st.chat_input("Que cherchez-vous ?")
if question:
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche des événements..."):
            reponse = chatbot(question)
        st.write(reponse)