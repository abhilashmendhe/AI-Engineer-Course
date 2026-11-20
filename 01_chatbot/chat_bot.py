import os 
from dotenv import load_dotenv
from openai import OpenAI

def bot(message: str, messages: list, llm_client: OpenAI):
    messages.append(
        {
            "role": "user",   # A role could be 'user', 'system', or 'assistant'
            "content": message
        }
    )
    response = llm_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        # temperature=1, # How deterministic we want a model
        # top_p=1,       
        # max_completion_tokens=2048,
        # reasoning_effort="medium",
        # stream=True,
        # stop=None
    )
    answer = response.choices[0].message.content
    messages.append({
        "role": "assistant",
        "content": answer
    })
    return answer

def main(llm_client: OpenAI):
    messages = [] # This is conversation history
    while True:
        user_input = input("Ask something (type 'q' to terminate): ").strip().lower()
        # print(messages)
        if user_input == 'q':
            print("Goodbye... 👋")
            break
        bot_answer = bot(user_input, messages, llm_client)
        print(f"Answer: {bot_answer}")
    
if __name__ == "__main__": 
    load_dotenv()
    groq_api_key = os.environ.get("GROQ_API_KEY")
    llm_client = OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key
    )
    main(llm_client)