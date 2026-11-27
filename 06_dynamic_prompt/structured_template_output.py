import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class User(BaseModel):
    name: str = Field(description="The name of the user")
    age: int = Field(description="Age of the user")
    companies: list[str] = Field(description="The companies associated with the user")
    net_worth: str = Field(description="The networth of the user")


class Agent:
    def __init__(self, model: str, temperature: float):
        self.llm = ChatGroq(model=model, temperature=temperature)

    def chat(self):
        structured_llm = self.llm.with_structured_output(User, method="json_schema")
        response = structured_llm.invoke(
            "Elon Reeve Musk (/ˈiːlɒn/ ⓘ; born June 28, 1971) is an industrialist who is the chief executive officer (CEO) and largest shareholder of Tesla and SpaceX. He has been the wealthiest person in the world since 2025, becoming the only trillionaire in history (in terms of US dollars) in 2026; as of November 2026, Forbes estimates his net worth to be $849.7 billion."
        )
        print(response)


if __name__ == "__main__":
    dotenv.load_dotenv()
    model_name = "openai/gpt-oss-20b"

    agent = Agent(model=model_name, temperature=0.5)
    agent.chat()
