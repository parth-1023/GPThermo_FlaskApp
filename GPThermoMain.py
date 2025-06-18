# GPThermoMain.py

import os
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Union
from GPThermoCoder import ThermoSolver
from langchain_openai import ChatOpenAI
from time import sleep

from dotenv import load_dotenv
load_dotenv() 
openai_api_key = os.getenv("OPENAI_API_KEY") 
print(f"OpenAI API Key: {openai_api_key}")

# ------------------ State ------------------
class GraphState(TypedDict, total=False):
    messages: list[Union[HumanMessage, AIMessage]]
    answer: str
    problem: str
    chat_response: str

# ------------------ Agent Class ------------------
class ConversationAgent:
    def __init__(self):
        self.chat_history = InMemoryChatMessageHistory()
        self.llm = ChatOpenAI(model="gpt-4o")
        self.solver = ThermoSolver()
        self.workflow = self._build_graph()

    def _generate_problem_node(self, state: GraphState) -> GraphState:
        user_messages = [m.content for m in state["messages"] if isinstance(m, HumanMessage)]
        context = "\n".join(user_messages[-5:])

        instruction = (
            "You are a thermodynamics assistant helping write solver prompts.\n"
            "Given the last few user messages in a conversation, create a single standalone prompt that describes the problem clearly.\n"
            "Do not include commentary. Do not format. Just return the full prompt for the solver."
        )

        response = self.llm.invoke([
            SystemMessage(content=instruction),
            HumanMessage(content=context)
        ])

        problem_prompt = response.content.strip()
        return {
            "problem": problem_prompt,
            "messages": state["messages"] + [HumanMessage(content=problem_prompt)]
        }

    def _router(self, state: GraphState) -> str:
        last_user_msg = next((msg for msg in reversed(state["messages"]) if isinstance(msg, HumanMessage)), None)
        if not last_user_msg:
            return "chat"

        routing_prompt = [
            SystemMessage(content=(
                "You are a router in a thermodynamics tutor system.\n"
                "Your job is to classify the user's latest message:\n"
                "- Return 'solve' if it asks for a thermodynamic quantity (enthalpy, pressure, temperature, etc.) involving a specific state, equation, or requires code/figure.\n"
                "- Return 'chat' if it's just a theory explanation, clarification, or conversational message.\n"
                "Respond with ONLY one of: solve or chat."
            )),
            last_user_msg
        ]

        response = self.llm.invoke(routing_prompt).content.strip().lower()
        return response if response == "solve" else "chat"

    def _solve_node(self, state: GraphState) -> GraphState:
        try:
            prompt = state.get("problem")
            if not isinstance(prompt, str) or not prompt.strip():
                return {"messages": state["messages"] + [AIMessage(content="[ERROR] No valid problem description found.")]}

            result = self.solver.solve(prompt)
            if isinstance(result, dict):
                code_output = result.get("code") or result.get("answer") or str(result)
            else:
                code_output = str(result)

            return {
                "answer": code_output,
                "messages": state["messages"] + [AIMessage(content=code_output)]
            }
        except Exception as e:
            return {
                "messages": state["messages"] + [AIMessage(content=f"[ERROR] {str(e)}")]
            }

    def _chat_node(self, state: GraphState) -> GraphState:
        if state.get("answer"):
            summary_prompt = f"The coder got the answer: {state['answer']} to the problem: {state['problem']}. Present the result to the user and explain it briefly."
            system_prompt = SystemMessage(content=summary_prompt)
        else:
            system_prompt = SystemMessage(content="You are a friendly and expert thermodynamics tutor. Respond to the student's message and help them understand the topic.")

        response = self.llm.invoke([system_prompt])
        return {
            "chat_response": response.content,
            "messages": state["messages"] + [response]
        }

    def _build_graph(self):
        graph = StateGraph(GraphState)
        graph.add_node("generate_problem", self._generate_problem_node)
        graph.add_node("solve", self._solve_node)
        graph.add_node("chat", self._chat_node)

        graph.add_conditional_edges(START, self._router, {
            "solve": "generate_problem",
            "chat": "chat"
        })
        graph.add_edge("generate_problem", "solve")
        graph.add_edge("solve", "chat")
        graph.add_edge("chat", END)

        return graph.compile()

    def run_conversation_chain(self, user_input: str) -> str:
        self.chat_history.add_user_message(user_input)
        result = self.workflow.invoke({"messages": self.chat_history.messages})

        sleep(3)
        
        final_messages = result["messages"]
        self.chat_history.messages.clear()
        for m in final_messages:
            if isinstance(m, HumanMessage):
                self.chat_history.add_user_message(m.content)
            elif isinstance(m, AIMessage):
                self.chat_history.add_ai_message(m.content)

        return final_messages[-1].content

# ------------------ Main Loop ------------------
if __name__ == "__main__":
    print("Welcome to GPThermo! Type your problem or ask a question.")
    agent = ConversationAgent()
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in {"exit", "quit"}:
                break
            response = agent.run_conversation_chain(user_input)
            # print(agent.chat_history)
            print("AI:", response)
        except Exception as e:
            print(f"[EXCEPTION] {e}")
