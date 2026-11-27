import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate


class Agent:
    def __init__(self, model: str, temperature: float):
        self.llm = ChatGroq(model=model, temperature=temperature)

    def chat(self):

        template = """Dear {user_name}
        Apologizing for the bad experience, we will set an appointment to 
        futher dicuss your problem.
        
        Sincerely 
        {brand_name}
        """
        prompt = PromptTemplate(
            input_variables=["user_name", "brand_name"], template=template
        )
        result = prompt.format(user_name="Alex", brand_name="NeonLearn")
        print(result)
        # response = self.llm.invoke("hello")
        # print(response.content)


if __name__ == "__main__":
    dotenv.load_dotenv()
    model_name = "openai/gpt-oss-20b"

    agent = Agent(model=model_name, temperature=0.5)
    agent.chat()
