import streamlit as st
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()


st.title("¡Bienvenido! Soy tu agente experto en matemáticas")


# Cargar PDF
pdf_file = st.file_uploader("Para ayudarte por favor sube tu PDF de matemáticas", type="pdf")
if pdf_file:

    file_path = os.path.join(pdf_file.name)
    with open (file_path, "wb") as f:
        f.write(pdf_file.getbuffer())

    loader = PyPDFLoader(pdf_file.name)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="llama3.2")
    vectorstore = FAISS.from_documents(splits, embeddings)

    retriever = vectorstore.as_retriever()
    llm = Ollama(model="llama3.2")

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print ("¡Perfecto! he procesado tu documento correctamente")

    query = st.text_input("Ahora escribe tu inquietud")
    if query:
        if "history" not in st.session_state:
            st.session_state.history = []

        with st.spinner("procesando"):
            respuesta = qa.run(query)

        st.session_state.history.append({"role": "user", "content": query})
        st.session_state.history.append({"role": "assistant", "content": respuesta})
    
    if "history" in st.session_state:
        for msg in st.session_state.history:
            if msg["role"] == "user":
                st.chat_message("user").write(msg["content"])
            else:
                st.chat_message("assistant").write(msg["content"])

    
    if st.button("Crear una guia educativa"):
         with st.spinner("procesando"):
             guia = qa.run(
                "Elabora una guía educativa basada en el documento. "
                "Incluye: definiciones clave, ejemplos claros y ejercicios prácticos."
             )

    st.write("## Comparto tu Guía Educativa.")
    st.write(guia)
    
# Para usar OpenIA
# openai_api_key = os.getenv("OPENAI_API_KEY")
# uploaded_file = st.file_uploader("Sube un PDF de matemáticas", type=["pdf"])
# if uploaded_file:
#     st.write("Procesando documento...")
#     loader = PyPDFLoader(uploaded_file.name)
#     docs = loader.load()

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_documents(docs)

#     embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
#     vectorstore = FAISS.from_documents(chunks, embeddings)

#     retriever = vectorstore.as_retriever()
#     qa_chain = RetrievalQA.from_chain_type(
#         llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
#         retriever=retriever
#     )

#     question = st.text_input("Escribe tu pregunta sobre el documento:")
#     if question:
#         answer = qa_chain.run(question)
#         st.markdown(f"**Respuesta:** {answer}")
