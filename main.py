import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("News Research Tool")
st.sidebar.title("News Article URLs")

urls = []
# urls = [
#     # "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html",
#     # "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html",
#     # "https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-743-kr-choksey-11080811.html",
#     "https://timesofindia.indiatimes.com/life-style/food-news/vada-pav-history-of-the-popular-mumbai-snack/articleshow/76973714.cms"
# ]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    print(loader)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    print("Data object length is: ", len(data))
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","], chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    print("Splitted chunks are: ", docs)

    # create embeddings and save it to FAISS index
    # OpenAI embeddings are a way to represent text data as vectors (lists of numbers).
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    if not embeddings:
        raise ValueError("Embeddings are empty")
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    print("Vectorstore processing is done.")
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index(is library which can be work as vector database of fetched articles)
    print("dimention of faiss during storing", vectorstore_openai.index.d)
    vectorstore_openai.save_local("faiss_store")
    print("Vectorstore is saved to file.")

query = main_placeholder.text_input("Question: ")
if query:
    vectorstore = FAISS.load_local(
        "faiss_store",
        OpenAIEmbeddings(model="text-embedding-3-large"),
        allow_dangerous_deserialization=True,
    )
    print("dimention of loaded faiss", vectorstore.index.d)
    print("Vectorstore is loaded from file.")
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever()
    )
    result = chain({"question": query}, return_only_outputs=True)
    # result will be a dictionary of this format --> {"answer": "", "sources": [] }
    st.header("Answer")
    st.write(result["answer"])

    # Display sources, if available
    sources = result.get("sources", "")
    if sources:
        st.subheader("Sources:")
        sources_list = sources.split("\n")  # Split the sources by newline
        for source in sources_list:
            st.write(source)
