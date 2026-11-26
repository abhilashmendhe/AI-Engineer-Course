import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from tools import get_weather
from langchain_tavily import TavilySearch

class Agent:
    def __init__(self, model: str, temperature: float):
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.search_tool = TavilySearch(max_results=1)
        self.messages = [
            SystemMessage("Give short answers, use the provided tools whenever needed")
        ]

        self.agent = create_agent(model=self.llm, tools=[get_weather, self.search_tool])

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

            response = self.agent.invoke({"messages": self.messages})

            self.messages.append(response['messages'][-1])

            print(response['messages'][-1].content)


if __name__ == "__main__":
    dotenv.load_dotenv()
    model_name = "openai/gpt-oss-20b"
    
    agent = Agent(model=model_name, temperature=0.5)
    agent.chat()
