# app.py
from flask import (
    Flask, request, jsonify,
    render_template, send_from_directory
)
from GPThermoMain import ConversationAgent
import os, pathlib

# -------------------------------------------------------------------
# basic config
# -------------------------------------------------------------------
STATIC_DIR   = pathlib.Path(__file__).parent / "static"
FIGURE_NAME  = "output.png"           # ThermoSolver should overwrite static/output.png

app   = Flask(__name__, static_folder=str(STATIC_DIR))
agent = ConversationAgent()

# -------------------------------------------------------------------
# serve the current diagram (timestamp in URL busts browser cache)
# -------------------------------------------------------------------
@app.route("/figure/<path:dummy>")
def serve_figure(dummy):
    return send_from_directory(
        app.static_folder, FIGURE_NAME, mimetype="image/png"
    )

# -------------------------------------------------------------------
# pages & API
# -------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/send_message", methods=["POST"])
def send_message():
    message = request.form.get("message", "").strip()
    if not message:
        return jsonify({"error": "Message is required"}), 400

    try:
        bot_message = agent.run_conversation_chain(message)
        return jsonify({"bot_message": bot_message})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

# -------------------------------------------------------------------
# WSGI entry-point
# -------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
