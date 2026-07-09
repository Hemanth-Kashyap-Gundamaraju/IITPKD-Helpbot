import os
from flask import Flask, request, jsonify
from backend import chain_builder

app = Flask(__name__)

# Initialize the lightweight cloud connection chain ONCE on boot
rag_chain = chain_builder.build_rag_chain()

@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    """Handles incoming payload requests from WhatsApp API webhooks."""
    data = request.get_json()
    
    try:
        # Extract messages from Meta WhatsApp API structural layouts
        message_body = data['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
        sender_phone = data['entry'][0]['changes'][0]['value']['messages'][0]['from']
        
        # Invoke inference
        response = rag_chain.invoke(message_body)
        answer = response.content[0]['text'] if isinstance(response.content, list) else response.content
        
        # TODO: Insert code snippet here to POST the answer back to Meta API
        print(f"Reply sent to {sender_phone}: {answer}")
        
        return jsonify({"status": "success"}), 200
    except Exception:
        # Return 200 to acknowledge receipt to Meta's servers even on empty events
        return jsonify({"status": "ignored"}), 200

if __name__ == "__main__":
    app.run(port=5000)