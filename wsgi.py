import os
import requests
from flask import Flask, request, jsonify
from backend import chain_builder

app = Flask(__name__)

# Load secret infrastructure tokens from environment
WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN")

rag_chain = chain_builder.build_rag_chain()

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """Handles the initial handshake verification with Meta Developer platform."""
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    """Handles real-time incoming messages, processes RAG, and replies via Meta API."""
    data = request.get_json()
    
    try:
        # Extract message body and user phone number
        message_value = data['entry'][0]['changes'][0]['value']
        if 'messages' in message_value:
            message_body = message_value['messages'][0]['text']['body']
            sender_phone = message_value['messages'][0]['from']
            
            # 1. Run inference through Groq + Pinecone
            response = rag_chain.invoke(message_body)
            answer = response.content[0]['text'] if isinstance(response.content, list) else response.content
            
            # 2. POST the answer back to Meta API to send the WhatsApp text
            url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
            headers = {
                "Authorization": f"Bearer {WHATSAPP_TOKEN}",
                "Content-Type": "application/json"
            }
            payload = {
                "messaging_product": "whatsapp",
                "to": sender_phone,
                "type": "text",
                "text": {"body": answer}
            }
            
            requests.post(url, json=payload, headers=headers)
            
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error handling webhook payload: {e}")
        return jsonify({"status": "ignored"}), 200