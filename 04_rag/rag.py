import time
import hashlib
import os
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


def read_pdf_file(file_name):
    docs = []

    reader = PdfReader(file_name)
    for ind, page in enumerate(reader.pages):
        doc = Document(
            page_content=page.extract_text() or "",
            metadata={"source": file_name, "page": ind + 1},
        )
        docs.append(doc)
    return docs


def generate_id(docs):
    return [hashlib.md5(doc.page_content.encode()).hexdigest() for doc in docs]


def llm(query, context, client):
    system_prompt = f"""You are a medical information assistant. Answer the user's question using ONLY
    the provided source documents below.

      Rules you must follow:
      1. Base your answer ONLY on the provided sources — do not use outside knowledge
      2. If the sources don't contain enough information to answer, say exactly: "I don't have enough information to answer
      this question."
      3. Cite sources using [Source N] notation whenever you use information from them
      4. If sources contradict each other, mention the contradiction
      5. Be concise — answer in 3-5 sentences unless the question requires more detail
      6. Never speculate or make assumptions beyond what the sources say

      Sources:
      {context}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b", temperature=0.5, messages=messages
    )

    return response.choices[0].message.content


def embed_docs(
    splitter: RecursiveCharacterTextSplitter, vector_store: Chroma, client: OpenAI
):
    chunks = splitter.split_documents(docs)

    # texts = [chunk.page_content for chunk in chunks]
    # vectors = model.embed_documents(texts) # model is HuggingFaceEmbeddings

    ids = generate_id(chunks)

    vector_store.add_documents(chunks, ids=ids)

    # all_docs = vector_store.get()
    while True:
        user_input = input("\nAsk something (type 'q' to terminate): ").strip().lower()
        if user_input == "q":
            print("Goodbye... 👋")
            break
        retrieved_chunks = vector_store.similarity_search_with_score(user_input, k=3)
        parts = []

        for i, (doc, score) in enumerate(retrieved_chunks):
            parts.append(
                f"Source {i + 1} | page {doc.metadata.get('page', '?')} | cosine similarity {score: .3f} \n {doc.page_content}"
            )
            # print(f"---- chunk {i} ----\n")
            # print(f"Score: {chunk[1]:.3f}")
            # print(f"Metadata:\n  - source: {chunk[0].metadata['source']}\n  - page: {chunk[0].metadata['page']}")
            # print(f"{chunk[0].page_content[:200]}\n")
        context = "\n\n_____\n\n".join(parts)
        # print(context)
        print(llm(user_input, context, client))


if __name__ == "__main__":
    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_request_url = "https://api.groq.com/openai/v1"
    # hf_model = "meta-llama/Llama-3.1-8B"
    hf_model = "NousResearch/Meta-Llama-3.1-8B"
    hf_token = os.environ.get("HF_TOKEN")

    # start = time.time()

    docs = read_pdf_file("./symptoms.pdf")

    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True},
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=110,
        chunk_overlap=10,
    )
    vector_store = Chroma(
        collection_name="medical_data",
        embedding_function=model,
        persist_directory="./chroma",
        collection_metadata={"hnsw:space": "cosine"},
    )

    client = OpenAI(base_url=groq_request_url, api_key=groq_api_key)

    # end = time.time()
    # print(f"Total time taken to create model, splitter, and vector_store: {end - start} s.")
    embed_docs(splitter, vector_store, client)
