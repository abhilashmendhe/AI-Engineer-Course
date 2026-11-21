import os
import time
from copy import deepcopy
from transformers import AutoTokenizer
from dotenv import load_dotenv
from openai import OpenAI


class ChatBot:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        hf_model: str,
        hf_token: str,
        max_number_trim_messages: int = 4,
    ):
        self.llm_client = OpenAI(base_url=base_url, api_key=api_key)

        self.messages = [
            {
                "role": "system",
                "content": """
                    You are a helpful assistant. Your name is Goku bot. 
                    Introduce yourself in the very first message.
                    Give short and helpful answers.
                """,
            }
        ]  # This is conversation history

        self.actual_history_messages = deepcopy(self.messages)
        self.tokenizers = AutoTokenizer.from_pretrained(hf_model, token=hf_token)

        self.total_tokens = 0
        self.input_tokens = 0
        self.output_tokens = 0

        self.max_number_trim_messages = max_number_trim_messages

    def trim_messages(self):
        if len(self.messages) < self.max_number_trim_messages:
            return 
        
        system_messages = [m for m in self.messages if m['role'] == 'system']
        other_messages = [m for m in self.messages if m['role'] != 'system']
        
        trimmed_messages = other_messages[-self.max_number_trim_messages:]
        
        self.messages = [*system_messages, *trimmed_messages]

    def bot(self, message: str, stream=False):
        self.messages.append(
            {
                "role": "user",  # A role could be 'user', 'system', or 'assistant'
                "content": message,
            }
        )
        self.actual_history_messages.append(
            {
                "role": "user",  # A role could be 'user', 'system', or 'assistant'
                "content": message,
            }
        )
        if not stream:
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=self.messages,
                # temperature=1, # How deterministic we want a model
                # top_p=1,
                # max_completion_tokens=2048,
                # reasoning_effort="medium",
                stream=stream,
                # stop=None
            )
            answer = response.choices[0].message.content
            self.total_tokens += response.usage.total_tokens
            self.input_tokens += response.usage.prompt_tokens
            self.output_tokens += response.usage.completion_tokens
        else:
            response = self.llm_client.chat.completions.create(
                model="openai/gpt-oss-20b",
                messages=self.messages,
                stream=stream,
            )
            answer = ""
            for chunk in response:
                if chunk.choices:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        answer += delta
                        print(f"{delta}", end="", flush=True)
                        time.sleep(0.05)

                if chunk.usage:
                    self.total_tokens += chunk.usage.total_tokens
                    self.input_tokens += chunk.usage.prompt_tokens
                    self.output_tokens += chunk.usage.completion_tokens
        # print()
        self.messages.append({"role": "assistant", "content": answer})
        self.actual_history_messages.append({"role": "assistant", "content": answer})
        self.trim_messages()
        if not stream:
            return answer

    def chat(self, stream=False):
        # encoding = tiktoken.get_encoding('cl100k_base')
        while True:
            user_input = (
                input("Ask something (type 'q' to terminate): ").strip().lower()
            )
            # print(messages)
            if user_input == "q":
                print("Goodbye... 👋")
                break

            inp_user_token_count = len(self.tokenizers.encode(user_input))
            if inp_user_token_count > 15:
                print(
                    "Your prompt exceeded the limit. Make sure your prompt words are around 12-14 words."
                )
                continue

            if self.total_tokens > 10000:
                print("You've exceeded the token limit.")
                break

            bot_answer = self.bot(user_input, stream)
            print(f"Answer: {bot_answer}\n")

    def report(self):
        print("\n---- #################### Printing Report #################### ----\n")
        print("\nConversation history of messages: ")
        for message in self.actual_history_messages:
            print(message)

        print(f"\nTotal input tokens: {self.input_tokens}")
        print(f"Total output tokens: {self.output_tokens}")
        print(f"Total tokens used: {self.total_tokens}")

        print(
            f"\nCost: $ {((self.input_tokens * 0.05 + self.output_tokens * 0.08) / 1e6):.8f} per million tokens."
        )


if __name__ == "__main__":
    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_request_url = "https://api.groq.com/openai/v1"
    # hf_model = "meta-llama/Llama-3.1-8B"
    hf_model = "NousResearch/Meta-Llama-3.1-8B"
    hf_token = os.environ.get("HF_TOKEN")
    print("\n -------------------- STARTED CHATBOT ---------------------- \n")
    chatbot = ChatBot(groq_request_url, groq_api_key, hf_model, hf_token)
    chatbot.chat(True)
    chatbot.report()
