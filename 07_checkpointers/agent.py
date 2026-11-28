import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver


class Agent:
    def __init__(self, model: str, temperature: float):
        self.llm = ChatGroq(model=model, temperature=temperature)

        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="Give short answers",
            checkpointer=InMemorySaver(),
        )

    def chat(self):

        while True:
            user_input = (
                input("\nAsk something (type 'q' to terminate): ").strip().lower()
            )
            if user_input == "q":
                print("Goodbye... 👋")
                break

            config = {"configurable": {"thread_id": "session_1"}}

            response = self.agent.invoke({"messages": [HumanMessage(content=user_input)]}, config)

            print(response["messages"][-1].content)


if __name__ == "__main__":
    dotenv.load_dotenv()
    model_name = "openai/gpt-oss-20b"

    agent = Agent(model=model_name, temperature=0.5)
    agent.chat()
