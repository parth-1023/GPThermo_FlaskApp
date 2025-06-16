import json
import logging

from GPThermoMain import ConversationAgent
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory

# configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# single, long‐lived agent in container
conversation_agent = ConversationAgent()

def handler(event, context):
    """
    Expects a JSON body with:
      {
        "message": "Your new user message",
        "chat_history": [
          {"role":"user","content":"..."},
          {"role":"assistant","content":"..."},
          …
        ]
      }
    Returns:
      {
        "response": "AI's reply",
        "chat_history": [ ... updated array ... ]
      }
    """
    try:
        body = json.loads(event.get("body", "{}"))
        user_message = body.get("message", "").strip()
        incoming = body.get("chat_history", [])

        # Reconstruct the LangChain history
        history = InMemoryChatMessageHistory()
        for msg in incoming:
            role = msg.get("role")
            text = msg.get("content", "")
            if role == "user":
                history.add_user_message(text)
            elif role == "assistant":
                history.add_ai_message(text)

        # Swap in the reconstructed history
        conversation_agent.chat_history = history

        # Run one turn
        ai_reply = conversation_agent.run_conversation_chain(user_message)

        # Build out the new history array
        out = []
        for m in conversation_agent.chat_history.messages:
            out.append({
                "role": "user" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            })

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps({
                "response": ai_reply,
                "chat_history": out
            })
        }

    except Exception as e:
        logger.exception("Error in Lambda handler")
        return {
            "statusCode": 500,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "OPTIONS,POST",
            },
            "body": json.dumps({"error": str(e)})
        }
