import os
import re
import requests
import threading  # <-- 1. Import threading to handle background tasks
from flask import Flask, request, jsonify
from backend import chain_builder

app = Flask(__name__)

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN")

rag_chain = chain_builder.build_rag_chain()

def process_ai_and_reply(message_body, sender_phone):
    """This runs in the background, allowing the webhook to reply 200 OK instantly."""
    try:
        # 1. Run inference through Groq + Pinecone (takes 1-3 seconds)
        response = rag_chain.invoke(message_body)
        answer = response.content[0]['text'] if isinstance(response.content, list) else response.content
        
        # 2. Strip the thinking process tags
        clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        
        # 3. POST the clean answer back to Meta API
        url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json"
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": sender_phone,
            "type": "text",
            "text": {"body": clean_answer}
        }
        
        requests.post(url, json=payload, headers=headers)
    except Exception as e:
        print(f"Error processing AI background response: {e}")

@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    data = request.get_json()
    
    try:
        message_value = data['entry'][0]['changes'][0]['value']
        if 'messages' in message_value:
            message_body = message_value['messages'][0]['text']['body']
            sender_phone = message_value['messages'][0]['from']
            
            # ⚡ SPIN OFF PROCESS AS A BACKGROUND THREAD & RETURN 200 OK IMMEDIATELY
            threading.Thread(target=process_ai_and_reply, args=(message_body, sender_phone)).start()
            
        return jsonify({"status": "success"}), 200  # <-- Returns in milliseconds!
    except Exception as e:
        print(f"Error handling webhook payload: {e}")
        return jsonify({"status": "ignored"}), 200