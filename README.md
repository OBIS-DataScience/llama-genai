# LangChain Question Answering with Llama 2 Model

## Overview

This project demonstrates the power of LangChain's Question Answering system using the Open Source Llama 2 Model from Facebook AI. It combines a variety of components to create a robust and efficient system for answering questions in a natural language context.

## Components

### Vector Stores

LangChain employs FAISS, a powerful vector database, for efficient storage and retrieval of text embeddings. Documents are loaded using the `UnstructuredFileLoader` from the `langchain.document_loaders` module. The documents are then split into manageable text chunks for optimal processing.

### Embeddings

To enhance the understanding of text and context, the project utilizes the 'sentence-transformers/all-MiniLM-L6-v2' model for generating embeddings. These embeddings are crucial for matching queries with relevant documents. The embeddings are computed with a specific device, such as 'cuda,' to maximize performance.

### Language Model

The core of the system is powered by the 'meta-llama/Llama-2-7b-chat-hf' model, fine-tuned for chat-based language understanding. It is loaded with several optimized settings, including:
- The use of float16 data types for improved efficiency.
- 8-bit quantization to further enhance performance while maintaining accuracy.
- An auth token for secure access to the model.

### Question Answering Pipeline

The 'HuggingFacePipeline' is a versatile tool that integrates the language model and tokenizer for text generation. It is configured with parameters that influence the generation process, including:
- Temperature: A parameter that controls the randomness of the generated text.
- Max token limit: The maximum number of tokens generated in a response.
- Top-k sampling: A technique for selecting the most likely tokens in the generated text.
- Number of return sequences: Controlling how many responses are generated.
- End-of-sequence token: Ensures that responses are appropriately terminated.

### Retrieval Question Answering Chain

The 'RetrievalQA' chain combines the components mentioned above into a coherent system. This chain is designed to effectively handle questions and retrieve context from the vector store. It is capable of providing detailed and relevant answers by matching queries with stored documents.

## Usage

You can use this LangChain Question Answering system by providing your query as the `query` variable in the script. The system will then generate responses based on the model's understanding and the document context stored in the vector store.

Example usage:

```python
query = "What is the capital of France?"
result = chain({"query": query}, return_only_outputs=True)
