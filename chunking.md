# Five Levels of Chunking Strategies in RAG| Notes from Greg’s Video

> Your goal is not to chunk for chunking sake, our goal is to get our data in a
> format where it can be retrieved for value later.

# Level 1 : Fixed Size Chunking

This is the most crude and simplest method of segmenting the text. It breaks
down the text into chunks of a specified number of characters, regardless of
their content or structure.

_Langchain and llamaindex_ framework offer
[CharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/character_text_splitter)
and
[SentenceSplitter](https://docs.llamaindex.ai/en/stable/api/llama_index.node_parser.SentenceSplitter.html)
(default to spliting on sentences) classes for this chunking technique. A few
concepts to remember -

- **How the text is split**: by single character
- **How the chunk size is measured:** by number of characters
- **chunk\_size:** the number of characters in the chunks
- **chunk\_overlap:** the number of characters that are being overlap in
  sequential chunks. keep duplicate data across chunks
- **separator**: character(s) on which the text would be split on (default “”)

# Level 2: Recursive Chunking

While Fixed size chunking is easier to implement, it doesn’t consider the
structure of text. Recursive chunking offers an alternative.

In this method, we divide the text into smaller chunk in a hierarchical and
iterative manner using a set of separators. If the initial attempt at splitting
the text doesn’t produce chunks of the desired size, the method recursively
calls itself on the resulting chunks with a different separator until the
desired chunk size is achieved.

Langchain framework offers
[RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)
class, which splits text using
[default separators](https://github.com/langchain-ai/langchain/blob/9ef2feb6747f5a69d186bd623b569ad722829a5e/libs/langchain/langchain/text_splitter.py#L842)
(“\\n\\n”, “\\n”, “ “,””)

# Level 3 : Document Based Chunking

In this chunking method, we split a document based on its inherent structure.
This approach considers the flow and structure of content but may not be as
effective documents lacking clear structure.

1. **Document with Markdown:** Langchain provides MarkdownTextSplitter class to
   split document that consist markdown as way of separator.
2. **Document with Python/JS:** Langchain provides PythonCodeTextSplitter to
   split the python program based on class, function etc. and We can
   [provide language into from\_language method](https://github.com/langchain-ai/langchain/blob/9ef2feb6747f5a69d186bd623b569ad722829a5e/libs/langchain/langchain/text_splitter.py#L983)
   of RecursiveCharacterTextSplitter class.
3. **Document with tables:** When dealing with tables, splitting based on levels
   1 and 2 might lose the tabular relationship between rows and columns. To
   preserve this relationship, format the table content in a way that the
   language model can understand (e.g., using `<table>` tags in HTML, CSV format
   separated by ';', etc.). During semantic search, matching on embeddings
   directly from the table can be challenging. Developers often summarize the
   table after extraction, generate an embedding of that summary, and use it for
   matching.
4. **Document with images (Multi- Modal):** Embeddings for images and text could
   be contents different (Though [CLIP](https://openai.com/research/clip) model
   support this). The ideal tactic is to use multi-modal model (like GPT-4
   vision) to generate summaries of the images and store embeddings of it.
   [Unstructured.io](https://unstructured-io.github.io/unstructured/introduction.html)
   provides partition\_pdf method to extract images from pdf document.

# Level 4: Semantic Chunking

All above three levels deals with content and structure of documents and
necessitate maintaining constant value of chunk size. This chunking method aims
to extract semantic meaning from embeddings and then assess the semantic
relationship between these chunks. The core idea is to keep together chunks that
are semantic similar.

![](https://cdn-images-1.readmedium.com/v2/resize:fit:800/1*C_pdPTvljd-nF2NLC0Bn7g.png)

Llamindex has
[SemanticSplitterNodeParse class](https://docs.llamaindex.ai/en/stable/examples/node_parsers/semantic_chunking.html)
that allows to split the document into chunks using contextual relationship
between chunks. This adaptively picks the breakpoint in-between sentences using
embedding similarity.

few concepts to know

- buffer\_size: configurable parameter that decides the initial window for
  chunks
- breakpoint\_percentile\_threshold: another configurable parameter. The
  threshold value to decide where to split the chunk
- embed\_mode: the embedding model used.

# Level 5: Agentic Chunking

This chunking strategy explore the possibility to use LLM to determine how much
and what text should be included in a chunk based on the context.

To generate initial chunks, it uses
[concept of Propositions based on paper](https://arxiv.org/pdf/2312.06648.pdf)
that extracts stand alone statements from a raw piece of text. Langchain
provides
[propositional-retrieval template](https://templates.langchain.com/new?integration_name=propositional-retrieval)
to implement this.

After generating propositions, these are being feed to LLM-based agent. This
agent determine whether a proposition should be included in an existing chunk or
if a new chunk should be created.

![](https://cdn-images-1.readmedium.com/v2/resize:fit:800/1*aHXJ5wuWuh1faf_BF7i4og.png)

Propositions are being created or included in existing chunk
