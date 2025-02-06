# LLM
1. Definition
LLM (Large Language Model) is a deep learning model that has been pre-trained on massive text datasets. The core of LLM is the Transformer architecture, consisting of encoder and decoder with self-attention mechanism (allowing transformer to consider different levels of importance between tokens when making predictions). RNN (Recurrent Neural Network) was the initial foundation for sequence data processing. Transformer improved upon RNN through the attention mechanism (introduced in Google's "Attention Is All You Need" paper in 2017), enabling more efficient processing. Modern LLMs are all based on Transformer architecture and no longer use RNN (evolution: RNN → Transformer → LLM (large scale)).

1. Origin
The fundamental tasks of LLM include Next Word Prediction, Text Generation, Text Completion, and Machine Translation. After being trained on large datasets, LLMs can perform Question Answering, Text Summarization, Reasoning, Code Generation, etc. The first organization to release a transformer model version was OpenAI with GPT-1 (Generative Pre-trained Transformer - 1), and with the advancement of specialized hardware like GPUs, the training and inference process of LLMs has been accelerated.

1. Types of LLMs:
- Pre-training models
- Fine-tuning models (Using a pre-trained model and adjusting it for a specific task - Example: Fine-tuning BERT for text classification, question answering)
- Multimodal models (capable of processing different types of data simultaneously - Example: GPT-4V can process both text and images)

