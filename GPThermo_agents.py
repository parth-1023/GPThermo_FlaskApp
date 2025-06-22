#GPThermo_agents.py

import os
import json
import pyromat as pm
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import StructuredTool
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from pydantic import BaseModel
from difflib import get_close_matches
from dotenv import load_dotenv
import numpy as np
from module_assistant import ModuleAgent
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

class ConversationAgent:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize PyroMat units
        pm.config['unit_pressure'] = 'MPa'
        pm.config['unit_temperature'] = 'K'
        pm.config['unit_energy'] = 'kJ'
        pm.config['unit_matter'] = 'kg'
        pm.config['unit_volume'] = 'm3'

        # Load fluid data from JSON file
        with open('PyromatInfo.json', 'r') as f:
            self.fluid_data = json.load(f)
        self.fluid_ids = {entry['ID']: entry['name'] for entry in self.fluid_data}

        # Initialize ModuleAgent
        self.module_assistant = ModuleAgent(module_folder="LearningModules/ES3001")

        # Define tools
        self.tools = self.define_tools()

        # Define the model
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

        # Define the agent's prompt with additional instructions for unit conversion.
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You can use select_fluid_id to find the PyroMat fluid ID of a fluid. When solving for work, use change in enthalpy. "
                "You can use solve_state and solve_process to solve thermodynamic states and processes. You can also use request_module to ask the module assistant a question. "
                "Let the student answer before solving; give hints and help students if they are stuck. Follow all instructions in modules. "
                "Present the initial question, and then move step by step through sub-questions. Present the entire question at each step. "
                "Let the student answer before solving; give hints and help students if they are stuck. Follow all instructions in modules. "
                "Ask the student to answer the questions before solving. The units for calculation functions are MPa, K, kJ, kg, and m3 by default. You will have to convert back and forth from these units. "
                "**Important:** Before solving, always verify that the input values are in the correct units. Use the update units tool to update pyromat units when needed"
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the tool-calling agent
        self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)

        # Create the executor
        self.executor = AgentExecutor(agent=self.agent, tools=self.tools)

        # Initialize chat history as a list of message objects
        self.chat_history = []

    def postMessage(self, message):
        """
        Helper function that ensures a message (or any object) is converted to a valid string.
        If the message is a dict or list, it converts it to a JSON string.
        Otherwise, it uses str() conversion.
        """
        if isinstance(message, (dict, list)):
            return json.dumps(message)
        return str(message)

    def define_tools(self):
        # Tool functions
        def guess_fluid_id_ai(user_input):
            prompt = (
                "You are tasked with identifying the correct PyroMat fluid based on user input.\n"
                "The following is a list of fluids and their descriptions: \n"
                f"{json.dumps(self.fluid_data)}\n"
                f"User Input: '{user_input}'\n"
                "If a fluid can be reasonably inferred (e.g., 'steam' is 'water'), provide the PyroMat fluid ID.\n"
                "If uncertain, make an educated guess.\n"
                "Example Output: ig.F2Fe\n"
                "Example Input: 'steam'\n"
                "Example Output: 'ig.H2O'\n"
                "The only content of the response is the PyroMat ID, nothing else.\n"
            )
            response = self.llm.invoke(prompt)
            response_text = response.content.strip()

            if response_text in self.fluid_ids:
                return response_text
            close_matches = get_close_matches(response_text, self.fluid_ids.keys(), n=1, cutoff=0.5)
            if close_matches:
                return close_matches[0]
            raise ValueError(f"No matching fluid found for input: {user_input}")

        def solve_state(**arguments):
            fluid = pm.get(arguments['working_fluid'])
            state = fluid.state(**{
                arguments['property_name_1']: arguments['property_value_1'],
                arguments['property_name_2']: arguments['property_value_2']
            })
            return {key: float(value) if np.isscalar(value) else float(value.item()) for key, value in state.items()}

        def solve_process(**arguments):
            fluid = pm.get(arguments['working_fluid'])
            initial_state = fluid.state(**{
                arguments['state1_property_name_1']: arguments['state1_property_value_1'],
                arguments['state1_property_name_2']: arguments['state1_property_value_2']
            })
            constant_property_name = {
                "isothermal": 'T',
                "isobaric": 'p',
                "isentropic": 's',
                "isochoric": 'v',
                "isenthalpic": 'h'
            }.get(arguments['process_type'])

            if not constant_property_name:
                raise ValueError(f"Unknown process type: {arguments['process_type']}")

            final_state = fluid.state(**{
                arguments['state2_property_name']: arguments['state2_property_value'],
                constant_property_name: initial_state[constant_property_name]
            })
            return {key: float(value) for key, value in final_state.items()}

        # New tool function to update PyroMat units based on user input
        def update_units(**arguments):
            if 'unit_pressure' in arguments and arguments['unit_pressure'] is not None:
                pm.config['unit_pressure'] = arguments['unit_pressure']
            if 'unit_temperature' in arguments and arguments['unit_temperature'] is not None:
                pm.config['unit_temperature'] = arguments['unit_temperature']
            if 'unit_energy' in arguments and arguments['unit_energy'] is not None:
                pm.config['unit_energy'] = arguments['unit_energy']
            if 'unit_matter' in arguments and arguments['unit_matter'] is not None:
                pm.config['unit_matter'] = arguments['unit_matter']
            if 'unit_volume' in arguments and arguments['unit_volume'] is not None:
                pm.config['unit_volume'] = arguments['unit_volume']
            return "PyroMat units updated successfully."

        # Argument schemas
        class requestModuleArgs(BaseModel):
            query: str

        class SelectFluidIdArgs(BaseModel):
            user_input: str

        class SolveStateArgs(BaseModel):
            working_fluid: str
            property_name_1: str
            property_value_1: float
            property_name_2: str
            property_value_2: float

        class SolveProcessArgs(BaseModel):
            process_type: str
            working_fluid: str
            state1_property_name_1: str
            state1_property_value_1: float
            state1_property_name_2: str
            state1_property_value_2: float
            state2_property_name: str
            state2_property_value: float

        class UpdateUnitsArgs(BaseModel):
            unit_pressure: str = None
            unit_temperature: str = None
            unit_energy: str = None
            unit_matter: str = None
            unit_volume: str = None

        # Define tools using StructuredTool
        return [
            StructuredTool(
                name="select_fluid_id",
                func=guess_fluid_id_ai,
                description="Guess the PyroMat fluid ID based on user input.",
                args_schema=SelectFluidIdArgs
            ),
            StructuredTool(
                name="solve_state",
                func=solve_state,
                description=(
                    "Solve thermodynamic state properties. Use the select_fluid_id tool to get the fluid ID first.\n"
                    "Enter the fluid ID in the working_fluid field ex. ig.H2O. Use units of K, MPa, kg/m3, kJ/kg, kJ/kgK.\n"
                    "Valid property names are T, p, d, v, cp, cv, gam, e, h, s, mw, R."
                ),
                args_schema=SolveStateArgs
            ),
            StructuredTool(
                name="solve_process",
                func=solve_process,
                description=(
                    "Solve thermodynamic processes. Use the select_fluid_id tool to get the fluid ID first.\n"
                    "Enter the fluid ID in the working_fluid field ex. ig.H2O. Use units of K, MPa, kg/m3, kJ/kg, kJ/kgK.\n"
                    "Valid property names are T, p, d, v, cp, cv, gam, e, h, s, mw, R."
                ),
                args_schema=SolveProcessArgs
            ),
            StructuredTool(
                name="load_module",
                func=self.module_assistant.load_module_and_generate_message,
                description="Select the most relevant learning module based on user input and load system instructions.",
                args_schema=requestModuleArgs
            ),
            StructuredTool(
                name="update_units",
                func=update_units,
                description=(
                    "Update the PyroMat units configuration. Provide any or all of unit_pressure, "
                    "unit_temperature, unit_energy, unit_matter, and unit_volume to update the PM configuration."
                ),
                args_schema=UpdateUnitsArgs
            )
        ]

    def handle_query(self, user_input, verbose=False):
        try:
            # Append user input as a HumanMessage
            self.chat_history.append(HumanMessage(content=user_input))
            max_iterations = 5  # Limit iterations to avoid infinite loops
            final_response = None

            # Prepare conversation input with a simple list of message contents
            conversation_input = {
                "input": user_input,
                "chat_history": [msg.content for msg in self.chat_history]
            }

            for _ in range(max_iterations):
                response = self.executor.invoke(conversation_input)
                # Convert the response to a valid string using postMessage
                response_str = self.postMessage(response)
                # Append the agent's response as an AIMessage with validated string content
                self.chat_history.append(AIMessage(content=response_str))

                # Check if the response appears final
                if self._is_final_response(response_str):
                    final_response = response_str
                    break

                conversation_input = {
                    "input": "",
                    "chat_history": [msg.content for msg in self.chat_history]
                }

            if final_response is None:
                final_response = response_str

            if verbose:
                return {
                    "input": user_input,
                    "output": final_response,
                    "chat_history": [self.postMessage(msg.content) for msg in self.chat_history]
                }
            else:
                return final_response

        except Exception as e:
            return f"Error: {str(e)}"

    def _is_final_response(self, response):
        """
        Checks whether the agent's response appears to be final.
        For example, if the response does not contain any tool name, it is considered final.
        """
        tool_names = ["select_fluid_id", "solve_state", "solve_process", "load_module", "update_units"]
        return not any(tool_name in response for tool_name in tool_names)


if __name__ == "__main__":
    conversation_agent = ConversationAgent()

    while True:
        user_input = input("Enter your query (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        response = conversation_agent.handle_query(user_input)
        # Print only the output from the verbose response if available
        if isinstance(response, dict):
            print(response.get("output", ""))
        else:
            print(response)
