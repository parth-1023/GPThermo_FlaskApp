# app.py
from flask import Flask, render_template, request, jsonify
import os
import sys
import json # Import the json module

# Add the directory containing GPThermo_agents.py and module_assistant.py to the Python path
# This assumes they are in the same directory as app.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from GPThermo_agents import ConversationAgent

app = Flask(__name__)

# Initialize the ConversationAgent.
# This should ideally be done once when the app starts, not per request.
# Make sure PyromatInfo.json and LearningModules/ES3001 exist in the same directory as this script.
try:
    conversation_agent = ConversationAgent()
    print("ConversationAgent initialized successfully.")
except Exception as e:
    print(f"Error initializing ConversationAgent: {e}")
    conversation_agent = None # Set to None if initialization fails

@app.route('/')
def index():
    """Renders the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat messages from the frontend and returns bot responses."""
    if conversation_agent is None:
        return jsonify({'response': 'Chatbot is not initialized. Check server logs for errors.'}), 500

    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'response': 'No message provided.'}), 400

    try:
        # Pass verbose=True to get the structured response including chat history
        bot_response_data = conversation_agent.handle_query(user_message, verbose=True)
        print(f"Bot response data: {bot_response_data}") # Debugging

        # The 'output' field from handle_query is currently a JSON string itself.
        # We need to parse it to get the actual desired string.
        raw_output = bot_response_data.get("output")
        bot_reply = "I encountered an issue processing your request." # Default error message

        if isinstance(raw_output, str):
            try:
                # Attempt to parse the raw_output as JSON
                parsed_output = json.loads(raw_output)
                # If successful, check if the parsed_output is a dictionary with an 'output' key
                if isinstance(parsed_output, dict) and 'output' in parsed_output:
                    bot_reply = parsed_output['output']
                else:
                    # If it's a JSON string but not the expected dictionary, use it as is
                    bot_reply = raw_output # Fallback to using the raw string
            except json.JSONDecodeError:
                # If raw_output is a string but not valid JSON, use it directly
                bot_reply = raw_output
        else:
            # If raw_output is not a string (e.g., already a dict from handle_query),
            # then we use it directly as it was previously intended.
            # This handles cases where GPThermo_agents.py might return a direct dict
            # in the future or for specific types of responses.
            if isinstance(raw_output, dict) and 'output' in raw_output:
                bot_reply = raw_output['output']
            else:
                bot_reply = str(raw_output) # Fallback to string conversion if it's some other type

        # Return only the extracted bot_reply to the frontend
        return jsonify({'response': bot_reply})
    except Exception as e:
        print(f"Error handling query: {e}")
        return jsonify({'response': f"An error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure necessary dummy files and directories exist for demonstration
    if not os.path.exists('PyromatInfo.json'):
        with open('PyromatInfo.json', 'w') as f:
            f.write('[\n  {"ID": "ig.H2O", "name": "water"},\n  {"ID": "ig.AIR", "name": "air"}\n]')
        print("Created dummy PyromatInfo.json")

    module_dir = 'LearningModules/ES3001'
    os.makedirs(module_dir, exist_ok=True)
    if not os.path.exists(os.path.join(module_dir, 'example_module.yaml')):
        with open(os.path.join(module_dir, 'example_module.yaml'), 'w') as f:
            f.write("""
module_title: Example Thermodynamics Module
instructions: Follow these steps to solve the problem.
problem statement: Calculate the efficiency of an ideal Rankine cycle.
questions:
  - part: What is the first step in analyzing a Rankine cycle?
    hints:
      - Consider the components of the cycle.
  - part: How do you determine the enthalpy at different states?
    hints:
      - Use the solve_state tool with given properties.
""")
        print(f"Created dummy {module_dir}/example_module.yaml")

    print("Starting Flask app...")
    app.run(debug=True) # debug=True is good for development, set to False for production
