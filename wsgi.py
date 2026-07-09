import os
import re
import requests
from flask import Flask, request, jsonify
from backend import chain_builder

app = Flask(__name__)

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN")

rag_chain = chain_builder.build_rag_chain()

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
            
            print(f"Processing incoming text synchronously: {message_body}")
            
            # 1. Force execution inside the main thread loop (Blocks till finished)
            response = rag_chain.invoke(message_body)
            answer = response.content[0]['text'] if isinstance(response.content, list) else response.content
            
            # 2. Strip out the thinking tags
            clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
            
            # 3. POST the clean text back to the Meta API
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
            
            res = requests.post(url, json=payload, headers=headers)
            print(f"Meta Send Status: {res.status_code}, Payload Response: {res.text}")
            
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error handling webhook payload: {e}")
        return jsonify({"status": "ignored"}), 200