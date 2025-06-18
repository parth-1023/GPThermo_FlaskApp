# app.py
from flask import Flask, request, jsonify, render_template
from web_thermo_agent import WebConversationAgent # Import your new agent file
import os

app = Flask(__name__)

# Initialize the agent once when the app starts
# For a high-traffic app, you might use a connection pool or per-request initialization
web_agent = WebConversationAgent()

# Ensure the static directory exists for serving images
static_dir = os.path.join(os.path.dirname(__file__), 'static')
os.makedirs(static_dir, exist_ok=True) # Create 'static' folder if it doesn't exist

@app.route('/')
def index():
    """Serves the main HTML page for the chat interface."""
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_thermo():
    """API endpoint to receive user questions and return AI responses."""
    user_input = request.json.get('question')
    if not user_input:
        return jsonify({"error": "No question provided"}), 400

    # Call the web-ready agent's method
    response_data = web_agent.run_conversation_chain(user_input)

    # response_data will be a dictionary: {"text_answer": "...", "figure_url": "..."}
    return jsonify(response_data)

if __name__ == '__main__':
    # Run the Flask app
    # debug=True allows for auto-reloading and better error messages during development
    app.run(debug=True)
