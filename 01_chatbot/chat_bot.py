import os 
from dotenv import load_dotenv
from openai import OpenAI

def bot(message: str, llm_client: OpenAI):
    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": message
            }
        ]
    )
    answer = response.choices[0].message.content
    return answer

def main(llm_client: OpenAI):
    user_input = input("Ask something: ")
    bot_answer = bot(user_input, llm_client)
    print(f"Answer: {bot_answer}")
    
if __name__ == "__main__": 
    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key
    )
    main(llm_client)