import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


class Agent:
    def __init__(self, model: str, temperature: float):
        self.llm = ChatGroq(model=model, temperature=temperature)

    def chat(self):

        history = [
            HumanMessage("I am tired"),
            AIMessage(">>>Ich bin müde<<<"),
            HumanMessage("I hate you"),
            AIMessage(">>>Ich hasse dich<<<"),
        ]
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "you are a translator from {origin} to {to}"),
                MessagesPlaceholder(variable_name="conversation_history"),
                ("human", "{input}"),
            ]
        )
        chain = prompt | self.llm  # single runnable chain

        response = chain.invoke(
            {
                "origin": "English",
                "conversation_history": history,
                "to": "German",
                "input": "I love you",
            }
        )

        print(response.content)


if __name__ == "__main__":
    dotenv.load_dotenv()
    model_name = "openai/gpt-oss-20b"

    agent = Agent(model=model_name, temperature=0.5)
    agent.chat()
