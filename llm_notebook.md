# LLM
1. Definition
- LLM (Large Language Model) is a deep learning model that has been pre-trained on massive text datasets. The core of LLM is the Transformer architecture, consisting of encoder and decoder with self-attention mechanism (allowing transformer to consider different levels of importance between tokens when making predictions). RNN (Recurrent Neural Network) was the initial foundation for sequence data processing. Transformer improved upon RNN through the attention mechanism (introduced in Google's "Attention Is All You Need" paper in 2017), enabling more efficient processing. Modern LLMs are all based on Transformer architecture and no longer use RNN (evolution: RNN → Transformer → LLM (large scale)).

1. Origin
- The fundamental tasks of LLM include Next Word Prediction, Text Generation, Text Completion, and Machine Translation. After being trained on large datasets, LLMs can perform Question Answering, Text Summarization, Reasoning, Code Generation, etc. The first organization to release a transformer model version was OpenAI with GPT-1 (Generative Pre-trained Transformer - 1), and with the advancement of specialized hardware like GPUs, the training and inference process of LLMs has been accelerated.

1. Types of LLMs:
- Pre-training models
- Fine-tuning models (Using a pre-trained model and adjusting it for a specific task - Example: Fine-tuning BERT for text classification, question answering)
- Multimodal models (capable of processing different types of data simultaneously - Example: GPT-4V can process both text and images)


# RAG
- Definition: A technique that retrieves relevant information from knowledge bases to enhance LLM-generated responses
- Motivation: Addresses hallucination issues commonly found in models like LLama-2 and ChatGPT
- Core concept: Retrieves document chunks based on query similarity using dense vector representations

## 1. Components and Architecture

### 1.1 Key Components
- Embeddings Model
  - Converts text into multi-dimensional vectors
  - Processes both documents and queries
- Vector Database (e.g., Pinecone)
  - Stores vector embeddings
  - Enables similarity search

### 1.2 Processing Pipeline
#### Document Processing
- Chunks larger documents into smaller segments
- Converts chunks to vector embeddings
- Stores vectors with metadata in database

#### Query Processing
- Converts user query to vector using same embeddings model
- Performs similarity search in vector database
- Retrieves most relevant document chunks

### 1.3 Context Window Management
- Query allocation: 10-20% of context window
- Retrieved chunks: 40-60% of context window
- Prompt template: 20-30% of context window
- Total length constraint (e.g., 4096 tokens for GPT-3.5)

## 2. Retrieval Techniques

### 2.1 Dense Retrieval
- Focus on semantic similarity
- Uses cosine similarity for comparison
- Based on vector representations

### 2.2 Sparse Retrieval
- Emphasizes exact lexical matching
- Keyword-based approach

### 2.3 Hybrid Retrieval
1. Initial filtering using Sparse Retrieval
2. Result refinement using Dense Retrieval

## 3. Advanced Techniques

### 3.1 Hybrid Search
- Combines semantic and keyword search
- Improves retrieval accuracy
- Utilizes language models for embedding conversion

### 3.2 Re-ranking
- Employs cross-encoders
- Refines initial search results

### 3.3 Chunking Strategies
- Critical impact on retrieval effectiveness
- Various methods for document segmentation