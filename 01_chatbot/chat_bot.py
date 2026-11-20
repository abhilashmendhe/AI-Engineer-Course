import os 
from dotenv import load_dotenv
from openai import OpenAI

class ChatBot:
    def __init__(self, base_url: str, api_key: str):
        self.llm_client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        self.messages = [
            {
                "role": "system",
                "content": """
                    You are a helpful assistant. Your name is Goku bot. 
                    Introduce yourself in the very first message.
                    Give short and helpful answers.
                """
            }    
        ] # This is conversation history
        
        self.total_tokens = 0
        
    def bot(self, message: str):
        self.messages.append(
            {
                "role": "user",   # A role could be 'user', 'system', or 'assistant'
                "content": message
            }
        )
        response = self.llm_client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=self.messages,
            # temperature=1, # How deterministic we want a model
            # top_p=1,       
            # max_completion_tokens=2048,
            # reasoning_effort="medium",
            # stream=True,
            # stop=None
        )
        answer = response.choices[0].message.content
        self.messages.append({
            "role": "assistant",
            "content": answer
        })
        self.total_tokens += response.usage.total_tokens
        return answer

    def chat(self):
        while True:
            user_input = input("Ask something (type 'q' to terminate): ").strip().lower()
            # print(messages)
            if user_input == 'q':
                print("Goodbye... 👋")
                break
            bot_answer = self.bot(user_input)
            print(f"Answer: {bot_answer}\n")
        
        
    def report(self):
        print("\n---- #################### Printing Report #################### ----\n")
        print("\nConversation history of messages: ")
        for message in self.messages:
            print(message)
        
        print(f"\nTotal tokens used: {self.total_tokens}\n")
    
if __name__ == "__main__": 
    
    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    groq_request_url = "https://api.groq.com/openai/v1"
    
    chatbot = ChatBot(groq_request_url, groq_api_key)
    chatbot.chat()
    chatbot.report()