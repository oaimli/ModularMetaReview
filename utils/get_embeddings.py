from openai import OpenAI
import time
import sys


def get_embeddings(sentences):
    client = OpenAI(api_key="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")
    embeddings = []
    for sentence in sentences:
        sentence_embedding = []
        while len(sentence_embedding) == 0:
            try:
                sentence_embedding = \
                    client.embeddings.create(input=[sentence], model="text-embedding-3-large").data[0].embedding
            except:
                error = sys.exc_info()[0]
                print("API error:", error)
                sentence_embedding = []
                time.sleep(1)
        embeddings.append(sentence_embedding)
    return embeddings

if __name__ == "__main__":
    sentences = ["I am as student.", "a b c d e"]
    print(get_embeddings(sentences))