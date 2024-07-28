
import streamlit as st
import torch
from llama_index.llms.huggingface import HuggingFaceLLM 
from llama_index.core.prompts.prompts import SimpleInputPrompt
from langchain.embeddings.huggingface import HuggingFaceEmbeddings 
from llama_index.core import ServiceContext, VectorStoreIndex, SimpleDirectoryReader
from llama_index.legacy.embeddings.langchain import LangchainEmbedding
import os

# Define the LLM and embedding models
system_prompt="""
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""
query_wrapper_prompt=SimpleInputPrompt("{query_str}")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="cpu",
    # uncomment this if using CUDA to reduce memory usage
    # model_kwargs={"torch_dtype": torch.float16 , "load_in_4bit":True}
)

embed_model=LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2"))

service_context=ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

# Streamlit app
st.title("Document Q&A Assistant")

# File upload
uploaded_files = st.file_uploader("Upload your documents", type=['txt', 'pdf'], accept_multiple_files=True)

if uploaded_files:
    # Save uploaded files
    doc_dir = "uploaded_docs"
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)

    for uploaded_file in uploaded_files:
        file_path = os.path.join(doc_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Load documents
    documents = SimpleDirectoryReader(doc_dir).load_data()

    # Create index
    index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    query_engine = index.as_query_engine()

    # Query input
    query = st.text_input("Enter your query")

    if query:
        # Get response
        response = query_engine.query(query)
        st.write("Response:", response)
