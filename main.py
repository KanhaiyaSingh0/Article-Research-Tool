import os
import streamlit as st
import langchain
from langchain.llms import GooglePalm
import pickle
import time
import networkx as nx
import sentence_transformers
#from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
#from langchain.chains.langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
#from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

load_dotenv()

st.title("Kanhaiya Research Tool")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)
process_url_clicked = st.sidebar.button("Process URLs")

file_path = "faiss_vector_index.pk1"

main_placeholder = st.empty()
llm = GooglePalm(temperature=0.2, max_tokens = 500)
if process_url_clicked:
    #load data
    loader = UnstructuredURLLoader(urls = urls)
    main_placeholder.text("Data Loading... Started...")
    data = loader.load()
    
    #split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators = ['\n\n', '\n', '-', ','],
        chunk_size = 1000
    )
    main_placeholder.text("Data spliting... Started...")
    docs = text_splitter.split_documents(data)
    
    #create embedding
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    
    # Create a FAISS instance for vector database from 'data' and save it to FAISS index
    vectordb = FAISS.from_documents(documents=docs,
                                 embedding=embeddings)
    
    main_placeholder.text("Embedding Vector Started Building...")
    #storing the vector index in the lical matchine
    with open(file_path, "wb") as f:
        pickle.dump(vectordb, f)
        
        
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=vectorstore.as_retriever(),
                                        input_key="question",
                                        return_source_documents=True
                                        #chain_type_kwargs={"prompt": PROMPT}
                                   )
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["result"])
            # st.header("Sources")
            # for doc in result["source_documents"]:
            #     st.write(doc.metadata["source"])
        
    







