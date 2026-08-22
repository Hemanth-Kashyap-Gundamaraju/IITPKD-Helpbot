import config
def build_whatsapp_send_url(phone_number_id):
    """
    Description: Builds the Meta Graph API URL for sending a WhatsApp
        message to one phone number. Pure string-building, no network
        call - split out so send_text() only has to worry about actually
        sending, not building the pieces.
    Inputs: phone_number_id (string). Reads global config.WHATSAPP_GRAPH_BASE_URL,
        config.WHATSAPP_GRAPH_API_VERSION.
    Outputs: returns a URL string. No globals changed.
    Dependencies: none.
    Utilities: called by WhatsAppSender.send_text().
    """
    return (
        f"{config.WHATSAPP_GRAPH_BASE_URL}/{config.WHATSAPP_GRAPH_API_VERSION}"
        f"/{phone_number_id}/messages"
    )


def build_whatsapp_headers(token):
    """
    Description: Builds the auth headers needed for a Meta Graph API call.
    Inputs: token (WhatsApp API bearer token). No globals read.
    Outputs: returns a dict. No globals changed.
    Dependencies: none.
    Utilities: called by WhatsAppSender.send_text().
    """
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def build_whatsapp_payload(to, text):
    """
    Description: Builds the JSON body Meta expects for a plain text
        WhatsApp message.
    Inputs: to (recipient phone number), text (message body). Reads
        global config.WHATSAPP_MESSAGE_TYPE.
    Outputs: returns a dict. No globals changed.
    Dependencies: none.
    Utilities: called by WhatsAppSender.send_text().
    """
    return {
        "messaging_product": "whatsapp",
        "to": to,
        "type": config.WHATSAPP_MESSAGE_TYPE,
        "text": {"body": text},
    }
