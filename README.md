# RAG_using_llama2
This repository implements a Retrieval-Augmented Generation (RAG) model using LLaMA 2, combining retrieval-based and generative approaches for improved response accuracy. It includes tools for data preprocessing, model training, and evaluation, ideal for customer support and conversational AI applications.

# Document Q&A with LLaMA-2

This project provides a web-based Q&A assistant that uses LLaMA-2, a Hugging Face model, to answer questions based on the content of uploaded documents. The interface is built using Streamlit, which allows for easy interaction and document management.

## Features

- Upload multiple documents (txt, pdf, docx)
- Use LLaMA-2 to answer questions based on the uploaded documents
- Clean up temporary files after processing

## Requirements

- Python 3.8 or higher
- Streamlit
- torch
- Hugging Face Transformers
- LangChain
- llama-index
- getpass

## Installation

1. Clone the repository: - update this

```bash
git clone https://github.com/nikhiljoshi7776/RAG_using_llama2.git
```

2. Install the required libraries:

```bash
pip install streamlit torch transformers einops accelerate langchain bitsandbytes llama-index llama-index-llms-llama-cpp llama-index-embeddings-huggingface llama-index-llms-huggingface langchain-community
```

## Usage

1. Set your Hugging Face API token:

```python
import os
from getpass import getpass

os.environ["HF_TOKEN"] = getpass("Enter your Hugging Face token: ")
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open your browser and go to `http://localhost:8501`. You will see an interface to upload your documents.

4. Upload the documents and ask questions about their content. The assistant will respond based on the context provided in the documents.

## Code Explanation

1. **Import Libraries**:
   - Import necessary libraries including Streamlit, torch, and components from `llama_index` and `langchain`.

2. **Set Hugging Face Token**:
   - Use `getpass` to securely input and set the Hugging Face API token.

3. **Define Prompts**:
   - Define system and query wrapper prompts for the Q&A assistant.

4. **Initialize Models**:
   - Initialize the Hugging Face LLaMA model and embedding model using specified parameters.

5. **Create Service Context**:
   - Create a service context with default settings, including chunk size, LLM, and embedding model.

6. **Streamlit UI**:
   - Set up Streamlit UI with a title and file upload interface.
   - Save uploaded files to a temporary directory.
   - Load documents from the temporary directory.
   - Create a vector store index from the loaded documents.
   - Create a query engine from the vector store index.
   - Provide an input field for users to ask questions and display the response.
   - Clean up temporary files after processing.

## Example

Here's a brief snippet of the core functionality:

```python
# Set the Hugging Face API token as an environment variable
os.environ["HF_TOKEN"] = getpass()

# Define the system prompt for the Q&A assistant
system_prompt = """
You are a Q&A assistant. Your goal is to answer questions as
accurately as possible based on the instructions and context provided.
"""

# Define the query wrapper prompt in the default format supportable by LLama2
query_wrapper_prompt = SimpleInputPrompt("{query_str}")

# Initialize the Hugging Face LLaMA model with specific parameters
llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.0, "do_sample": False},
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    tokenizer_name="meta-llama/Llama-2-7b-chat-hf",
    model_name="meta-llama/Llama-2-7b-chat-hf",
    device_map="cpu"
)

# Initialize the embedding model using Hugging Face embeddings
embed_model = LangchainEmbedding(
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
)

# Create a service context with default settings, including chunk size, LLM, and embedding model
service_context = ServiceContext.from_defaults(
    chunk_size=1024,
    llm=llm,
    embed_model=embed_model
)

# Streamlit UI setup
st.title("Document Q&A with LLama-2")
```

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

This project is licensed under the MIT License.
