# Context-Aware RAG Chatbot

A chatbot that answers questions about **artificial intelligence and machine learning** using **Retrieval-Augmented Generation (RAG)**: it first looks up the most relevant passages in a small knowledge base and then lets a language model write the answer from them, while remembering the last few turns of the conversation. It is built with **LangChain, FAISS and Streamlit** and is made to run in **Google Colab**. It was built as a Month 2 task of an AI internship.

## How it works

```text
Wikipedia pages -> split into chunks -> local embeddings -> FAISS index (saved to kb_faiss/)
                                                                  |
your question --> retriever (most similar chunks) --> ChatOpenAI --> answer
                        ^                                  |
                        +---- last 3 exchanges (memory) ---+
```

| Part | What is used |
|---|---|
| Knowledge base | The English Wikipedia articles for *Artificial intelligence*, *Machine learning*, *Natural language processing* and *Large language model* |
| Chunking | `RecursiveCharacterTextSplitter`, 600 characters per chunk with 50 overlap |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, run **locally and free** (no API key needed to build the index) |
| Vector store | FAISS, saved to the `kb_faiss/` folder |
| Chain | LangChain `ConversationalRetrievalChain` |
| Language model | OpenAI `gpt-3.5-turbo`, temperature 0.3 |
| Memory | `ConversationBufferWindowMemory` keeping the last 3 exchanges, so follow-up questions such as "and how is it different from that?" work |
| Interface | A Streamlit page with a question box |
| Public link | `localtunnel` shows the Streamlit page through a temporary web address from Colab |

Only the chat model needs a key. The Streamlit page asks you to type your OpenAI API key into a password box in the sidebar; the key is used for that session and is **not saved anywhere in this project**.

## Run it in Google Colab

1. Open `RAG_Chatbot.ipynb` in Colab.
2. Run the first cell (installs `wikipedia` and `sentence-transformers`).
3. Run the second cell. It downloads the four Wikipedia articles, builds the FAISS index in `kb_faiss/` (this also downloads the small embedding model) and writes the Streamlit app to `app_ui.py`.
4. Run the third cell. It starts Streamlit and `localtunnel` and prints a temporary `loca.lt` address. localtunnel asks for a "tunnel password": it is the **endpoint IP** the cell prints just before the address.
5. Open the address, paste your OpenAI API key in the sidebar and ask a question, for example "What is retrieval-augmented generation?" and then "How does it differ from fine-tuning?".

The notebook itself installs only `wikipedia` and `sentence-transformers`; the LangChain packages were already available in the Colab environment it was run in. To run it anywhere else, install `requirements.txt` first.

## Files

```text
RAG_Chatbot.ipynb   the whole chatbot: knowledge-base builder, app code, launch cells
requirements.txt    Python packages (versions are not pinned)
```

`kb_faiss/` and `app_ui.py` are created when the notebook runs and are not stored in this repository.

## Limitations

- **A small knowledge base.** Only four Wikipedia articles (one page per topic) are indexed, so questions outside those pages get weak or invented answers. The notebook does not show which passages an answer came from.
- **It needs an OpenAI key and internet access**, and each question uses paid API calls.
- The notebook still contains an older, unused cell (`build_knowledge_base` with OpenAI embeddings) that refers to a variable that is never defined; it is not part of the working flow.
- The saved index is loaded with `allow_dangerous_deserialization=True`. That is fine for an index you built yourself, but never load an index file from someone you do not trust.
- LangChain changes its interfaces often, so with other package versions some imports may need small updates.
