import os
import dotenv
from transformers import pipeline

# Using a small encoder-decoder model for one-shot sentiment analyzer. This model has small parameters that can be loaded
# on the memory efficiently. Also we don't need to make explicitly call like we do for LLMs that then generate 
# responses. Calls are made locally to BERT that generates responses.

class SentimentAnalyzer:
    def __init__(
        self,
    ):
        self.classifier = pipeline(
            "zero-shot-classification", model="typeform/distilbert-base-uncased-mnli"
        )

    def sentiment_analysis(self, text: str):
        labels = ["positive", "negative", "neutral"]

        result = self.classifier(text, candidate_labels=labels)

        return result

    def chat_with_analyzer(self):

        while True:
            text = input("Enter a text (type 'q' to terminate): ").strip().lower()
            if text == "q":
                break
            output = self.sentiment_analysis(text)
            print(output)
            print()


if __name__ == "__main__":
    s_analyzer = SentimentAnalyzer()
    s_analyzer.chat_with_analyzer()
