import os
import re
import requests
from concurrent.futures import ThreadPoolExecutor  # <-- 1. Switched to ThreadPoolExecutor
from flask import Flask, request, jsonify
from backend import chain_builder

app = Flask(__name__)

# Initialize a global thread pool executor
executor = ThreadPoolExecutor(max_workers=4)

WHATSAPP_TOKEN = os.environ.get("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.environ.get("WHATSAPP_PHONE_NUMBER_ID")
VERIFY_TOKEN = os.environ.get("WHATSAPP_VERIFY_TOKEN")

rag_chain = chain_builder.build_rag_chain()

def process_ai_and_reply(message_body, sender_phone):
    """This function is submitted to the executor pool to process in the background."""
    try:
        # 1. Run inference through Groq + Pinecone
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
        
        res = requests.post(url, json=payload, headers=headers)
        print(f"Meta API Response Code: {res.status_code}, Content: {res.text}")
        
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
            
            # ⚡ Use the executor pool to guarantee execution space on Render
            executor.submit(process_ai_and_reply, message_body, sender_phone)
            
        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error handling webhook payload: {e}")
        return jsonify({"status": "ignored"}), 200