import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


class Agent:
    def __init__(self, model: str, temperature: float):
        self.llm = ChatGroq(model=model, temperature=temperature)

        self.messages = [SystemMessage("Give short answers")]

    def chat(self):

        while True:
            user_input = (
                input("\nAsk something (type 'q' to terminate): ").strip().lower()
            )
            # print(messages)
            if user_input == "q":
                print("Goodbye... 👋")
                break

            self.messages.append(HumanMessage(user_input))

            response = self.llm.invoke(self.messages)

            self.messages.append(response)

            print(response.content)


if __name__ == "__main__":
    dotenv.load_dotenv()
    model_name = "openai/gpt-oss-20b"
    agent = Agent(model=model_name, temperature=0.5)
    agent.chat()