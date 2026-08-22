import httpx
from fastapi import FastAPI, Request, Query , Body, Header
from fastapi.responses import PlainTextResponse
from backend import chain_builder
from backend.utils.response_utils import clean_llm_response
import config
from whatsapp.whatsapp_sender import WhatsAppSender, extract_incoming_text_message

app = FastAPI()

VERIFY_TOKEN = config.WHATSAPP_VERIFY_TOKEN

rag_chain = chain_builder.build_rag_chain()

# One shared sender for the whole app, built once from config at import time.
whatsapp_sender = WhatsAppSender(config.WHATSAPP_TOKEN, config.WHATSAPP_PHONE_NUMBER_ID)


@app.get("/webhook")
async def verify_webhook(request: Request):
    """
    Description: Handles Meta's webhook verification handshake - Meta calls
        this once when the webhook URL is first configured, to confirm we
        own it.
    Inputs: request (FastAPI Request - its query params are read). Reads
        global VERIFY_TOKEN.
    Outputs: returns the challenge string (200) on success, or
        "Verification failed" (403) otherwise.
    Dependencies: none.
    Utilities: registered as the GET /webhook FastAPI route.
    """
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return PlainTextResponse(challenge, status_code=200)
    return PlainTextResponse("Verification failed", status_code=403)


@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    """
    Description: Handles an incoming WhatsApp message - answers it using
        the RAG chain and sends the reply back. Always returns a success
        status to Meta (even on internal errors) so Meta doesn't keep
        retrying the same webhook delivery. Runs async end-to-end: the
        LLM call and the WhatsApp send both happen with `await`, so this
        request doesn't block other incoming webhook calls while it waits.
    Inputs: request (FastAPI Request - its JSON body is read). Reads
        global rag_chain, _whatsapp_sender.
    Outputs: returns a dict (FastAPI turns it into JSON automatically).
    Dependencies: calls _extract_incoming_text_message(), rag_chain.ainvoke(),
        clean_llm_response(), _whatsapp_sender.send_text().
    Utilities: registered as the POST /webhook FastAPI route.
    """
    try:
        body = await request.json()
        incoming = extract_incoming_text_message(body)
        if incoming is None:
            return {"status": "success"}

        message_body, sender_phone = incoming
        print(f"Processing incoming text: {message_body}")

        response = await rag_chain.ainvoke(message_body)
        clean_answer = clean_llm_response(response)
        await whatsapp_sender.send_text(sender_phone, clean_answer)

        return {"status": "success"}
    except Exception as e:
        print(f"Error handling webhook payload: {e}")
        return {"status": "ignored"}

from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def call_chat_loop(payload: ChatRequest):
    try:
        print(f"Processing incoming text: {payload.message}")

        response = await rag_chain.ainvoke(payload.message)
        clean_answer = clean_llm_response(response)

        return {"status": "success", "answer": clean_answer}
    except Exception as e:
        print(f"Error handling chat request: {e}")
        return {"status": "error", "message": str(e)}