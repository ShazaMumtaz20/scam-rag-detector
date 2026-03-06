import streamlit as st
from rag import analyze_scam

st.title("AI Scam Detection Assistant")

msg = st.text_area("Paste suspicious message here:")

if st.button("Analyze"):
    if msg.strip() == "":
        st.write("Please enter a message!")
    else:
        result = analyze_scam(msg)
        st.write(result)
