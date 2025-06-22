#module_assitant.py
import os
import yaml
from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import SystemMessagePromptTemplate


class ModuleAgent:
    def __init__(self, module_folder: str = "LearningModules", llm_model: str = "gpt-4o-mini",
                 temperature: float = 0.7):
        """
        ModuleAgent searches for the most relevant learning module YAML file and loads it.
        It then generates a system message based on the module contents.
        """
        self.module_folder = module_folder
        self.llm = ChatOpenAI(model=llm_model, temperature=temperature)
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = self._load_modules()
        self.current_module: Optional[Dict[str, Any]] = None
        self.current_module_key: Optional[str] = None

    def _load_modules(self):
        """
        Indexes all YAML learning modules using FAISS for efficient retrieval.
        Skips any files that fail to parse.
        """
        docs = []
        module_files = [f for f in os.listdir(self.module_folder) if f.endswith('.yaml')]

        if not module_files:
            raise ValueError(f"No YAML files found in {self.module_folder}")

        for file in module_files:
            module_key = file.replace(".yaml", "")
            module_path = os.path.join(self.module_folder, file)

            try:
                with open(module_path, "r", encoding="utf-8") as f:
                    content = yaml.safe_load(f)

                # Ensure content is not empty
                if not content:
                    print(f"Skipping empty module file: {file}")
                    continue

                text = f"Module: {content.get('module_title', module_key)}\n" \
                       f"Instructions: {content.get('instructions', 'No instructions')}\n" \
                       f"Problem: {content.get('problem statement', 'No problem statement')}"
                docs.append(Document(page_content=text, metadata={"module_key": module_key}))

            except yaml.YAMLError as e:
                print(f"Error parsing {file}: {e}")
                continue  # Skip the broken file instead of crashing

        if not docs:
            raise ValueError("No valid learning modules found. Please check YAML formatting.")

        return FAISS.from_documents(docs, self.embeddings)

    def load_most_relevant_module(self, query: str) -> str:
        """
        Searches for the most relevant module based on user input and loads it as the current module.
        """
        results = self.vectorstore.similarity_search(query, k=1)
        if not results:
            return "No suitable module found."

        module_key = results[0].metadata["module_key"]
        module_file = os.path.join(self.module_folder, f"{module_key}.yaml")

        try:
            with open(module_file, "r", encoding="utf-8") as file:
                self.current_module = yaml.safe_load(file)
                self.current_module_key = module_key
                return f"Loaded module: {module_key}"
        except Exception as e:
            return f"Error loading module '{module_key}': {e}"

    def generate_system_message(self) -> str:
        """
        Generates a structured system message based on the currently loaded module.
        """
        if not self.current_module:
            return "No module loaded. Please load a module first."

        instructions = self.current_module.get("instructions", "No specific instructions provided.")
        problem_statement = self.current_module.get("problem statement", "No problem statement provided.")
        questions = self.current_module.get("questions", [])

        # Format questions
        formatted_questions = [
            {"part": q.get("part", "Unnamed Question"), "hints": q.get("hints", [])}
            for q in questions
        ]

        # Structure the YAML output
        system_message_dict = {
            "system_message": {
                "module_title": self.current_module.get("module_title", "Unknown Module"),
                "instructions": instructions,
                "problem_statement": problem_statement,
                "questions": formatted_questions,
                "guidelines": [
                    "Encourage the student to attempt each step.",
                    "Provide hints if they struggle. Offer hints but do not provide them unless requested.",
                    "Fully explain solutions to ensure understanding.",
                    "Summarize key concepts at the end.",
                    "Do not ask the user if they have additional information. Only use information provided in the module."
                ],
            }
        }

        return yaml.dump(system_message_dict, default_flow_style=False, allow_unicode=True)

    def load_module_and_generate_message(self, query: str) -> str:
        """
        Loads the most relevant module based on user input and generates a structured system message.
        """
        load_result = self.load_most_relevant_module(query)
        if "Error" in load_result or "No suitable module found" in load_result:
            return load_result  # Return the error message directly

        return self.generate_system_message()


if __name__ == "__main__":
    agent = ModuleAgent(module_folder="LearningModules/ES3001")

    user_query = "Rankine cycle efficiency"
    print(agent.load_most_relevant_module(user_query))
    print(agent.generate_system_message())
