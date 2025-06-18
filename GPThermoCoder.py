# _____________________ SETUP _____________________
import getpass, os, pyromat as pm
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from typing_extensions import TypedDict, Literal
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_community.document_loaders import OnlinePDFLoader
from unstructured.partition.pdf import partition_pdf
from io import BytesIO
import requests
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import pyromat as pm
import matplotlib
matplotlib.use('Agg')
import json
import base64
from pathlib import Path
from langchain_core.utils.function_calling import convert_to_openai_function
import logging
import traceback
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


logging.basicConfig(level=logging.INFO)

#default file for generated pngs
#figure_path = "output.png"
figure_path = Path(__file__).parent / "static" / "output.png"


# _____________________ SCHEMA _____________________
class ThermoProblem(BaseModel):
    problem: str = Field(
        ...,
        description="Extract the relavent information for solving the user's thermodynamics problem. This will go to a "
        "psuedocode agent so keep it as short as possible, only the info the psuedocode agent will need. " \
        "extract the units specified in the problem into the correct fields if a figure/graph/plot etc is needed, set the figure_required to true",
    )
    pseudocode: str = Field(
        ...,
        description=(
            "Concise, ordered pseudocode the coding agent can use to write Python code with pyromat to solve the problem. "
            "Must include: (1) the working fluid(s) and any property calls, (2) each specified thermodynamic process or "
            "state transformation with its required inputs, (3) intermediate calculations, and (4) the final output(s). "
            "Do not add theory explanations, or any other commentary. No unit conversions, and this will be solved with code "
            "so no tables. You can create graphs and figures with matplotlib if needed. Be specific about correct units, "
            "specify units to use for T, p, d, v, cp, cv, gam, e, h, s, mw, R. Write the correct units for the units fields."
        ),
    )
    unit_energy: str = Field(default="kJ", description="Extracted energy unit mentioned in the problem (e.g., 'kJ', 'J', etc). ")
    unit_matter: str = Field(default="kg", description="Extracted mass unit mentioned in the problem (e.g., 'kg', 'g').")
    unit_temperature: str = Field(default="K", description="Extracted temperature unit (e.g., 'C', 'K', 'F') from the problem.")
    unit_pressure: str = Field(default="MPa", description="Extracted pressure unit (e.g., 'kPa', 'bar', 'MPa', 'atm', 'pa) from the problem.")
    figure_required: bool = Field(default=False, description="True if the user asked for a figure, diagram, graph or plot (anytime matplotlib would be needed).")

# _____________________ STATE _____________________
class State(TypedDict, total=False):
    problem: str
    pseudocode: ThermoProblem
    documentation: str
    code: str
    answer: str
    figure: bool = False

# _____________________ SOLVER CLASS _____________________
class ThermoSolver:
    def __init__(self, model_name: str = "gpt-4o-2024-08-06"):
        os.environ["LANGSMITH_TRACING"] = "true"
        load_dotenv()

        with open('PyromatInfo.json', 'r') as f:
            self.fluid_data = json.load(f)
        self.fluid_ids = {entry['ID']: entry['name'] for entry in self.fluid_data}

        self.llm = ChatOpenAI(model=model_name, temperature=0.0)
        thermo_fn = convert_to_openai_function(ThermoProblem)
        self.psuedocode_llm = ChatOpenAI(
            model=model_name,
            temperature=0.0,
        ).bind_tools([ThermoProblem])

        self.coder_llm = ChatOpenAI(model=model_name, temperature=0.0)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

        self.docs = self._load_pyromat_docs()
        splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=600, separators=["\n\n", "\n", " ", ""])
        splits = splitter.split_documents(self.docs)
        self.vector_store = InMemoryVectorStore.from_documents(splits, self.embeddings)
        self.retriever = self.vector_store.as_retriever(k=3, search_type="mmr")

        self.coder_agent = create_python_agent(
            llm=self.coder_llm,
            tool=PythonREPLTool(),
            verbose=True,
            handle_parsing_errors=True,
        )

        self.chain = self.build_chain().compile()

    def _load_pyromat_docs(self):
        try:
            local_path = os.path.join(os.path.dirname(__file__), "pyromat_handbook.pdf")
            loader = PyPDFLoader(local_path)
            docs = loader.load()
            return [
                Document(
                    page_content=" ".join(doc.page_content.split()).strip(),
                    metadata={"source": "pyromat_handbook.pdf"}
                )
                for doc in docs
                if len(doc.page_content.strip()) > 50
            ]
        except Exception as e:
            print(f"PDF loading error: {str(e)}")
            return []

    def _get_documentation(self, query: str) -> str:
        return (
            "You are a documentation retrieval assistant. Given a thermodynamics problem or pseudocode, your goal is to return only the "
            "most relevant documentation excerpts from the Pyromat handbook that would help a coding agent solve the problem using pyromat and matplotlib. "
            "Prioritize examples of function usage, correct fluid IDs, syntax for unit configuration, and anything specific to the calculations the user must make." 
            "DO NOT include theory, history, or generic introductions.") \
            + "\n\n" + \
            "QUERY: " + query + "\n\n" + \
            "RESULTS:\n" + "\n\n".join([
                f"DOC EXCERPT ({i+1}):\n{d.page_content}" for i, d in enumerate(self.retriever.invoke(query))
            ])

    def _call_coder_agent(self, problem_obj: ThermoProblem) -> str:
        unit_energy = problem_obj.unit_energy
        unit_matter = problem_obj.unit_matter
        unit_temperature = problem_obj.unit_temperature
        unit_pressure = problem_obj.unit_pressure
        prompt = problem_obj.pseudocode

        result = self.coder_agent.invoke({
            "input": f"PYROMAT DOCUMENTATION:\n{self._get_documentation(prompt)}\n\n"
                      f"TASK:\n{prompt}\n\n"
                      f"Fluid ID information for reference: {json.dumps(self.fluid_data)}\n\n"
                      "Write code using pyromat. Print a single string with the answer(s) and units, as well as assumptions as if you were answering on a test."
                      "DO NOT generate any figures. Another agent will handle that if needed."

                      f"Use these pyromat units:\nEnergy: {unit_energy}, Matter: {unit_matter}, Temperature: {unit_temperature}, Pressure: {unit_pressure}\n"
                      "Set pyromat units like this:\n"
                      "```python\n"
                      "import pyromat as pm\n"
                      f"pm.config['unit_energy'] = '{unit_energy}'\n"
                      f"pm.config['unit_matter'] = '{unit_matter}'\n"
                      f"pm.config['unit_temperature'] = '{unit_temperature}'\n"
                      f"pm.config['unit_pressure'] = units.Pressure('{unit_pressure}')\n"
                      "```\n"
        })
        return result["output"]

    def _make_figure(self, state: "State", filepath: str = figure_path) -> State:
        try:
            problem_obj = self.thermo_problem
            unit_energy = problem_obj.unit_energy
            unit_matter = problem_obj.unit_matter
            unit_temperature = problem_obj.unit_temperature
            unit_pressure = problem_obj.unit_pressure
            prompt = problem_obj.pseudocode
            code_context = state.get("code", "")
            answer_context = state.get("answer", "")

            figure_prompt = (
                "Your job is to generate a P-V, T-s or relevant thermodynamic diagram using matplotlib and pyromat.\n"
                "Use the code and answer provided to visualize the thermodynamic state or process.\n"
                "You must:\n"
                "1. Load the correct pyromat fluid (e.g., 'mp.H2O')\n"
                "2. Recalculate or confirm values using pyromat.\n"
                "3. Use matplotlib to generate a figure.\n"
                f"4. Save it directly to the path: `{filepath}` as a PNG file.\n"
                "5. DO NOT return any base64 string, commentary, or output to chat. Just save the file.\n\n"

                "Set units like this:\n"
                "```python\n"
                "import pyromat as pm\n"
                f"pm.config['unit_energy'] = '{unit_energy}'\n"
                f"pm.config['unit_matter'] = '{unit_matter}'\n"
                f"pm.config['unit_temperature'] = '{unit_temperature}'\n"
                f"pm.config['unit_pressure'] = '{unit_pressure}'\n"
                "```\n\n"

                "Here is an example that saves a figure directly:\n"
                "```python\n"
                "import pyromat as pm\n"
                "import matplotlib.pyplot as plt\n"
                "pm.config['unit_energy'] = 'kJ'\n"
                "pm.config['unit_matter'] = 'kg'\n"
                "pm.config['unit_temperature'] = 'C'\n"
                "pm.config['unit_pressure'] = 'MPa'\n"
                "water = pm.get('mp.H2O')\n"
                "temps = list(range(100, 600, 10))\n"
                "p = 2\n"
                "s_vals = [water.s(T=t, p=p)[0] for t in temps]\n"
                "plt.figure()\n"
                "plt.plot(s_vals, temps, label=f'{p} MPa')\n"
                "plt.scatter(water.s(T=500, p=p)[0], 500, color='red')\n"
                "plt.xlabel('Entropy (kJ/kg·K)')\n"
                "plt.ylabel('Temperature (°C)')\n"
                "plt.title('T-s Diagram')\n"
                "plt.grid(True)\n"
                f"plt.savefig(r'{filepath}', format='png')\n"
                "```\n\n"

                "**DO NOT print or return anything but True or False. Just save the file to the specified path.**\n\n"
                f"==== CONTEXT ====\n\n"
                f"PROBLEM:\n{state['problem']}\n\n"
                f"CODE:\n{code_context}\n\n"
                f"ANSWER:\n{answer_context}\n\n"
                f"PYROMAT DOCS:\n{self._get_documentation(prompt)}\n\n"
                "IMPORTANT: if you create a figure, print True, else print False"
                "Do not print anything else.\n\n"
            )

            response = self.coder_agent.invoke({"input": figure_prompt})
            output_text = response.get("output", "").strip()

            if output_text == "True" and os.path.exists(filepath):
                return {"figure": True, "figure_path": filepath}
            else:
                return {"figure": False}


        except Exception as e:
            logging.error(f"Figure generation failed: {e}")
            return {}



    def _write_pseudocode(self, state: "State") -> "State":
        prob = state.get("problem")
        if not prob:
            return {"error": "No problem found in state."}
        try:
            ai_msg = self.psuedocode_llm.invoke(prob)
            logging.info(f"Raw tool call result: {ai_msg}")
            if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
                args = ai_msg.tool_calls[0]["args"]
                result = ThermoProblem(**args)
            else:
                raise ValueError("Invalid tool call result")
            self.thermo_problem = result
            return {"pseudocode": result}
        except Exception as e:
            traceback.print_exc()
            logging.error(f"Pseudocode generation failed: {e}")
            return {"error": f"pseudocode node failed: {e}"}

    def build_chain(self):
        workflow = StateGraph(State)

        workflow.add_node("pseudocoder", lambda s, self=self: self._write_pseudocode(s))
        workflow.add_node("coder", lambda s, self=self: {"code": self._call_coder_agent(s["pseudocode"])} if "pseudocode" in s else {"error": "Missing pseudocode from previous step"})
        workflow.add_node("figure_maker", lambda s, self=self: self._make_figure(s))

        workflow.add_edge(START, "pseudocoder")
        workflow.add_edge("pseudocoder", "coder")
        workflow.add_conditional_edges("coder", lambda s: s["pseudocode"].figure_required,{True: "figure_maker",False: END,})
        workflow.add_edge("figure_maker", END)
        return workflow
    

    def get_graph(self):
        return self.build_chain()  # return uncompiled graph


    def displayFigure(self, state: "State", figure_path: str = figure_path):
        if state.get("figure"):
            if os.path.exists(figure_path):
                print("Displaying generated figure:")
                img = mpimg.imread(figure_path)
                plt.imshow(img)
                plt.axis('off')
                plt.show()
            else:
                print("Figure flag was set but file not found.")
        else:
            print("No figure was generated.")

    def solve(self, problem: str):
        initial_state = {"problem": problem}
        response = self.chain.invoke(initial_state)
        #self.displayFigure(response)
        return response.get("code")

# _____________________ MAIN EXEC _____________________
if __name__ == "__main__":
    initial_state = {
        "problem": "plot an adiabatic proscess where water goes from 500c and 3mpa to 1atm"
    }
    solver = ThermoSolver()
    result = solver.solve(initial_state["problem"])
    print(result)
    print("UNITS USED:", solver.thermo_problem.unit_energy, ",", solver.thermo_problem.unit_matter, ",", solver.thermo_problem.unit_temperature, ",", solver.thermo_problem.unit_pressure)



