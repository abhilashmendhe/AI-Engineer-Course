import os
import dotenv
from openai import OpenAI


class SentimentAnalyzer:
    def __init__(
        self,
        base_url: str,
        api_key: str,
    ):
        self.llm_client = OpenAI(base_url=base_url, api_key=api_key)

        self.messages = [
            {
                "role": "system",
                "content": "Classify the sentiment of the text as 'positive', 'negative', or 'neutral'",
            },
            
            # Few shot prompting: Below we provide some examples to LLM to guide its response on a new task 
            # without retraining the model. 3-5 examples are enough.
            {
                "role": "user",
                "content": "I absolutely love this restraurant!"
            },
            {
                "role": "assitant",
                "content": "positive"
            },
            {
                "role": "user",
                "content": "the wait time was way too long"
            },
            {
                "role": "assitant",
                "content": "negative"
            },
            {
                "role": "user",
                "content": "it was an average experience, nothing special"
            },
            {
                "role": "assitant",
                "content": "neutral"
            },
        ]  # This is conversation history

    def sentiment_analysis(self, text: str):
        response = self.llm_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[*self.messages, {"role": "user", "content": text}],
            temperature=0.1,  # How deterministic we want a model
        )
        return response.choices[0].message.content

    def chat_with_analyzer(self):

        while True:
            text = input("Enter a text (type 'q' to terminate): ").strip().lower()
            if text == "q":
                break
            output = self.sentiment_analysis(text)
            print(output)
            print()


if __name__ == "__main__":
    dotenv.load_dotenv()

    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_request_url = "https://api.groq.com/openai/v1"
    # hf_model = "meta-llama/Llama-3.1-8B"
    hf_model = "NousResearch/Meta-Llama-3.1-8B"
    hf_token = os.environ.get("HF_TOKEN")

    s_analyzer = SentimentAnalyzer(base_url=groq_request_url, api_key=groq_api_key)
    s_analyzer.chat_with_analyzer()
