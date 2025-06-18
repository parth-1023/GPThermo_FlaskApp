# web_thermo_agent.py

import os
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Union
from GPThermoCoder import ThermoSolver # Assumes GPThermoCoder.py is in the same directory
from langchain_openai import ChatOpenAI
from time import sleep
import traceback # Import for better error logging in web context
from pathlib import Path # For handling figure paths

from dotenv import load_dotenv
load_dotenv()
# OpenAI API key is loaded here via dotenv, so it's available for ChatOpenAI

# ------------------ State (same as your GPThermoMain.py) ------------------
class GraphState(TypedDict, total=False):
    messages: list[Union[HumanMessage, AIMessage]]
    answer: str
    problem: str
    chat_response: str
    figure_generated: bool # Added for web context
    figure_path: str      # Added for web context

# ------------------ Agent Class (modified run_conversation_chain) ------------------
class WebConversationAgent: # Renamed to avoid conflict if you import both
    def __init__(self):
        # In a web app, you might want a fresh history per user session,
        # or manage it externally. For simplicity, this creates a new one each time.
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
            return "chat" # Default to chat if no human message

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

            # ThermoSolver.solve now returns a dict with 'code', 'figure', 'figure_path'
            solver_full_result = self.solver.solve(prompt)
            code_output = solver_full_result.get("code", "[ERROR] Solver did not return an answer.")
            
            # Extract figure info to pass to the chat node and final return
            figure_generated = solver_full_result.get("figure", False)
            figure_path_from_solver = solver_full_result.get("figure_path", "")

            return {
                "answer": code_output,
                "messages": state["messages"] + [AIMessage(content=code_output)],
                "figure_generated": figure_generated,
                "figure_path": figure_path_from_solver
            }
        except Exception as e:
            traceback.print_exc() # Log traceback for web debugging
            return {
                "messages": state["messages"] + [AIMessage(content=f"[ERROR] An error occurred during solving: {str(e)}")]
            }

    def _chat_node(self, state: GraphState) -> GraphState:
        # Get the original user message from the state
        user_message_content = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), "")

        system_prompt_content = "You are a friendly and expert thermodynamics tutor. Respond to the student's message and help them understand the topic."
        
        # Check if an answer was just provided by the solver node
        if state.get("answer"):
            solver_answer = state['answer']
            problem_description = state['problem']
            figure_generated = state.get("figure_generated", False)
            
            summary_parts = [
                f"The problem asked was: \"{problem_description}\"." if problem_description else "",
                f"The thermodynamic solver provided the following result: \"{solver_answer}\"."
            ]
            if figure_generated:
                summary_parts.append("A relevant thermodynamic diagram has also been generated and saved.")
            
            # Refine the prompt to ask the LLM to summarize/explain the solver's answer
            system_prompt_content = (
                "You are a friendly and expert thermodynamics tutor. "
                "Present the following information to the user in a clear, concise, and helpful manner. "
                "Do not add additional technical details beyond what is provided. Encourage further questions. "
                "\n\nContext from solver:\n" + "\n".join(filter(None, summary_parts)) # Filter out empty strings
            )
            
            # Pass the original user message as the human message to the LLM for context
            messages_for_llm = [
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=user_message_content)
            ]
        else:
            # For pure chat questions, just use the general tutor prompt and the user's message
            messages_for_llm = [
                SystemMessage(content=system_prompt_content),
                HumanMessage(content=user_message_content)
            ]

        try:
            response = self.llm.invoke(messages_for_llm)
            return {
                "chat_response": response.content,
                "messages": state["messages"] + [response]
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "chat_response": f"[ERROR] An error occurred in the chat response generation: {str(e)}",
                "messages": state["messages"] + [AIMessage(content=f"[ERROR] An error occurred in the chat response generation: {str(e)}")]
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

    def run_conversation_chain(self, user_input: str) -> dict:
        # Clear history for each new web request, acting as a stateless API endpoint
        # For stateful conversations, you'd manage history in Flask sessions or a DB
        self.chat_history.clear()
        self.chat_history.add_user_message(user_input)

        try:
            result = self.workflow.invoke({"messages": self.chat_history.messages})
        except Exception as e:
            traceback.print_exc()
            return {"error": f"Error invoking the conversation chain: {str(e)}"}

        final_answer_text = result["messages"][-1].content
        
        # Determine if a figure was generated and construct its URL
        figure_url = None
        if result.get("figure_generated", False) and result.get("figure_path"):
            # Ensure figure_path is relative to the static folder for web serving
            # ThermoSolver saves to `static/output.png`
            # Flask serves from `static/`
            # So, if ThermoSolver's figure_path is `some_dir/static/output.png`,
            # we just need `static/output.png` for the web URL.
            figure_url = os.path.join("static", os.path.basename(result["figure_path"]))

        return {
            "text_answer": final_answer_text,
            "figure_url": figure_url
        }
