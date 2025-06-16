# GPThermoMain.py
# import logging
# logging.basicConfig(
#     level=logging.DEBUG,
#     format="%(asctime)s %(levelname)s:%(name)s:%(message)s"
# )

import os
from os import getenv

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
        text = last_user_msg.content if last_user_msg else ""
        # logging.debug(f"Router sees message: “{text}”")
        # if not last_user_msg:
        #     return "chat"

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
        route = response if response == "solve" else "chat"
        # logging.debug(f"→ routing to: {route}")
        return route

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
        last_user_msg = next(
            (m for m in reversed(state["messages"])
             if isinstance(m, HumanMessage)),
            None
        )

        if state.get("answer"):
            summary_prompt = f"The coder got the answer: {state['answer']} to the problem: {state['problem']}. Present the result to the user and explain it briefly."
            system_prompt = SystemMessage(content=summary_prompt)
        else:
            system_prompt = SystemMessage(content="You are a friendly and expert thermodynamics tutor. Respond to the student's message and help them understand the topic.")

        messages = [system_prompt]
        if last_user_msg:
            messages.append(last_user_msg)

        response = self.llm.invoke(messages)

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
        # print(f"--- Turn Start ---")
        # print(f"User Input: {user_input}")

        # Add the new user message to the history BEFORE invoking the graph.
        self.chat_history.add_user_message(user_input)
        # print(f"Chat History AFTER adding user_input (before invoke): {[m.content for m in self.chat_history.messages]}")


        # Invoke the workflow with the current, complete chat history.
        result = self.workflow.invoke({"messages": self.chat_history.messages})

        # Remove the sleep(3) line in final code, it's just for demo.
        sleep(3)

        # The 'result["messages"]' now contains the full updated history from the graph run.
        # print(f"Chat History AFTER invoke (result['messages']): {[m.content for m in result['messages']]}")

        # IMPORTANT FIX: Re-initialize chat_history or carefully update it.
        # The previous method of direct assignment might not be fully flushing/updating
        # InMemoryChatMessageHistory's internal state depending on its implementation.
        # The safest way is to clear and re-add all messages from the result.
        # Alternatively, if you want to strictly use add_message for all updates,
        # you need to iterate through new messages *only*.

        # Let's try the most robust way: completely replace the history with the result's messages
        # This ensures self.chat_history matches what the graph just produced.
        # self.chat_history.messages.clear() # Clear the old history
        for msg in result["messages"]:     # Add all messages from the graph's final state
            # Add messages back, preserving their type (HumanMessage/AIMessage)
            if isinstance(msg, HumanMessage):
                self.chat_history.add_user_message(msg.content)
            elif isinstance(msg, AIMessage):
                self.chat_history.add_ai_message(msg.content)
            # Handle other message types if they appear (e.g., SystemMessage, ToolMessage)
            # For simplicity, if they aren't Human/AI, they won't be added to chat_history for subsequent turns
            # if they are intermediate, it's fine. If they are meant to persist, extend this logic.


        # print(f"Chat History self.chat_history.messages AFTER update: {[m.content for m in self.chat_history.messages]}")
        # print(f"--- Turn End ---")

        # The last message in the updated history is the final response.
        return self.chat_history.messages[-1].content

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
            print("AI:", response)
        except Exception as e:
            print(f"[EXCEPTION] {e}")
