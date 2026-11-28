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

        active_session = "session_1"

        while True:
            user_input = (
                input(
                    f"\n({active_session})Ask something (type 'q' to terminate or 'ch' to change the session): "
                )
                .strip()
                .lower()
            )
            if user_input == "q":
                print("Goodbye... 👋")
                break

            if user_input == "ch":
                active_session = (
                    "session_2" if active_session == "session_1" else "session_1"
                )
                print(f"Switched to session: {active_session}")
                continue

            config = {"configurable": {"thread_id": active_session}}

            response = self.agent.invoke(
                {"messages": [HumanMessage(content=user_input)]}, config
            )

            print(response["messages"][-1].content)


if __name__ == "__main__":
    dotenv.load_dotenv()
    model_name = "openai/gpt-oss-20b"

    agent = Agent(model=model_name, temperature=0.5)
    agent.chat()
