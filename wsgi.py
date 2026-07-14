import requests
from flask import Flask, request, jsonify
from backend import chain_builder
from backend.response_utils import clean_llm_response
import config

app = Flask(__name__)

VERIFY_TOKEN = config.WHATSAPP_VERIFY_TOKEN

rag_chain = chain_builder.build_rag_chain()


class WhatsAppSender:
    """
    Description: Sends a text message back to a WhatsApp user via the Meta
        Graph API. Groups together the auth token and phone number ID -
        the two pieces every send needs - as attributes on one object
        instead of reading them from module-level globals inside the
        function body (which would make this hard to test in isolation).
    Inputs (constructor): token (WhatsApp API bearer token), phone_number_id
        (the sending number's Meta phone number ID).
    Utilities: used by whatsapp_webhook().
    """

    def __init__(self, token, phone_number_id):
        self.token = token
        self.phone_number_id = phone_number_id

    def send_text(self, to, text):
        """
        Description: POSTs a plain text message to the Meta Graph API for
            one WhatsApp recipient. This is the only method in this file
            that makes a network call, so it's the only piece that needs
            a real (or mocked) connection to test.
        Inputs: to (recipient phone number), text (message body). Reads
            self.token, self.phone_number_id. Reads global
            config.WHATSAPP_GRAPH_BASE_URL, config.WHATSAPP_GRAPH_API_VERSION,
            config.WHATSAPP_MESSAGE_TYPE.
        Outputs: returns the requests.Response object. No globals changed.
        Dependencies: uses requests.post().
        Utilities: called by whatsapp_webhook().
        """
        url = (
            f"{config.WHATSAPP_GRAPH_BASE_URL}/{config.WHATSAPP_GRAPH_API_VERSION}"
            f"/{self.phone_number_id}/messages"
        )
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": to,
            "type": config.WHATSAPP_MESSAGE_TYPE,
            "text": {"body": text},
        }
        res = requests.post(url, json=payload, headers=headers)
        print(f"Meta Send Status: {res.status_code}, Payload Response: {res.text}")
        return res


# One shared sender for the whole app, built once from config at import time.
_whatsapp_sender = WhatsAppSender(config.WHATSAPP_TOKEN, config.WHATSAPP_PHONE_NUMBER_ID)


def _extract_incoming_text_message(payload):
    """
    Description: Pulls the message text and sender phone number out of a
        raw WhatsApp webhook payload, if the payload actually contains an
        incoming text message (WhatsApp also sends other event types -
        delivery receipts, read receipts - that this should ignore).
        Pure parsing logic with no network calls, so it's easy to unit
        test with a plain fake payload dict.
    Inputs: payload (dict - the raw JSON body from Meta). No globals read.
    Outputs: returns a (message_body, sender_phone) tuple, or None if this
        payload isn't an incoming text message. No globals changed.
    Dependencies: none.
    Utilities: called by whatsapp_webhook().
    """
    try:
        message_value = payload["entry"][0]["changes"][0]["value"]
        if "messages" not in message_value:
            return None

        message_body = message_value["messages"][0]["text"]["body"]
        sender_phone = message_value["messages"][0]["from"]
        return message_body, sender_phone
    except (KeyError, IndexError, TypeError):
        return None


@app.route("/webhook", methods=["GET"])
def verify_webhook():
    """
    Description: Handles Meta's webhook verification handshake - Meta calls
        this once when the webhook URL is first configured, to confirm we
        own it.
    Inputs: none (reads query params from the Flask request). Reads global
        VERIFY_TOKEN.
    Outputs: returns (challenge string, 200) on success, or
        ("Verification failed", 403) otherwise.
    Dependencies: none.
    Utilities: registered as the GET /webhook Flask route.
    """
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        return challenge, 200
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def whatsapp_webhook():
    """
    Description: Handles an incoming WhatsApp message - answers it using
        the RAG chain and sends the reply back. Always returns 200 to Meta
        (even on internal errors) so Meta doesn't keep retrying the same
        webhook delivery.
    Inputs: none (reads JSON body from the Flask request). Reads global
        rag_chain, _whatsapp_sender.
    Outputs: returns a (JSON, status code) Flask response.
    Dependencies: calls _extract_incoming_text_message(), rag_chain.invoke(),
        clean_llm_response(), _whatsapp_sender.send_text().
    Utilities: registered as the POST /webhook Flask route.
    """
    try:
        incoming = _extract_incoming_text_message(request.get_json())
        if incoming is None:
            return jsonify({"status": "success"}), 200

        message_body, sender_phone = incoming
        print(f"Processing incoming text synchronously: {message_body}")

        response = rag_chain.invoke(message_body)
        clean_answer = clean_llm_response(response)
        _whatsapp_sender.send_text(sender_phone, clean_answer)

        return jsonify({"status": "success"}), 200
    except Exception as e:
        print(f"Error handling webhook payload: {e}")
        return jsonify({"status": "ignored"}), 200