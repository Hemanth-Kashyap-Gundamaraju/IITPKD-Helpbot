from .whatsapp_utils import build_whatsapp_headers ,build_whatsapp_payload,build_whatsapp_send_url
import httpx
import config

class WhatsAppSender:
    """
    Description: Sends a text message back to a WhatsApp user via the Meta
        Graph API. Groups together the auth token and phone number ID -
        the two pieces every send needs - as attributes on one object.
        send_text() is now async so the server can handle other incoming
        webhook requests while waiting on Meta's response, instead of
        freezing up.
    Inputs (constructor): token (WhatsApp API bearer token), phone_number_id
        (the sending number's Meta phone number ID).
    Utilities: used by whatsapp_webhook().
    """

    def __init__(self, token, phone_number_id):
        self.token = token
        self.phone_number_id = phone_number_id

    async def send_text(self, to, text):
        """
        Description: POSTs a plain text message to the Meta Graph API for
            one WhatsApp recipient. This is the only method in this file
            that makes a network call, so it's the only piece that needs
            a real (or mocked) connection to test. Runs async so it
            doesn't block the rest of the server while waiting for Meta.
        Inputs: to (recipient phone number), text (message body). Reads
            self.token, self.phone_number_id.
        Outputs: returns the httpx.Response object. No globals changed.
        Dependencies: calls _build_whatsapp_send_url(), _build_whatsapp_headers(),
            _build_whatsapp_payload(); uses httpx.AsyncClient.
        Utilities: called by whatsapp_webhook().
        """
        url = build_whatsapp_send_url(self.phone_number_id)
        headers = build_whatsapp_headers(self.token)
        payload = build_whatsapp_payload(to, text)

        async with httpx.AsyncClient() as client:
            res = await client.post(url, json=payload, headers=headers)

        print(f"Meta Send Status: {res.status_code}, Payload Response: {res.text}")
        return res




def extract_incoming_text_message(payload):
    """
    Description: Pulls the message text and sender phone number out of a
        raw WhatsApp webhook payload, if the payload actually contains an
        incoming text message (WhatsApp also sends other event types -
        delivery receipts, read receipts - that this should ignore).
        Pure parsing logic with no network calls, so it's easy to unit
        test with a plain fake payload dict. Unchanged from the Flask
        version - this logic doesn't care which web framework calls it.
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

