import os
import dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langchain.agents.middleware import SummarizationMiddleware


class Agent:
    def __init__(self, model: str, temperature: float, checkpointer: PostgresSaver):
        self.llm = ChatGroq(model=model, temperature=temperature)

        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt="Give short answers",
            checkpointer=checkpointer,
            middleware=[
                SummarizationMiddleware(
                    model=self.llm, trigger=("tokens", 200), keep=("messages", 4)
                )
            ],
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

        config_history = {"configurable": {"thread_id": "session_3"}}

        state = self.agent.get_state(config_history)
        print(
            state.values["messages"]
        )  # prints the state of the whole config_history for only session_1


if __name__ == "__main__":
    dotenv.load_dotenv()
    db_url = os.environ["DB_URL"]

    model_name = "openai/gpt-oss-20b"

    with PostgresSaver.from_conn_string(db_url) as checkpointer:
        checkpointer.setup()

        agent = Agent(model=model_name, temperature=0.5, checkpointer=checkpointer)
        agent.chat()
