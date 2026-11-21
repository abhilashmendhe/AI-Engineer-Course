from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util


def find_closest(query: str, sentences: list, model):
    embeddings = model.encode(sentences)
    query_embeddings = model.encode(query)

    closet_index = 0
    closest_score = -1

    for i, vector in enumerate(embeddings):
        similarity = util.cos_sim(vector, query_embeddings).item()
        if similarity > closest_score:
            closest_score = similarity
            closet_index = i

    return sentences[closet_index]


if __name__ == "__main__":
    load_dotenv()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    sentences = [
        "That is a happy person",
        "That is a happy dog",
        "That is a very happy person",
        "Today is a sunny day",
    ]

    query = "I am feeling great today"
    closest_sentence = find_closest(query, sentences, model)
    print(closest_sentence)

    # using in-built util method `semantic_search`
    embeddings = model.encode(sentences)
    query_embeddings = model.encode(query)
    
    ans = util.semantic_search(query_embeddings, embeddings, top_k=2)    
    print(ans)