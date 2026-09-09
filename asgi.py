import httpx
from fastapi import FastAPI, Request, Query, Body, Header
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from backend import chain_builder
from backend.utils.response_utils import clean_llm_response
import config
from backend.utils.cache_utils import get_response_cache
from backend.utils.analytics import get_query_logger
from backend.utils.escalation import get_escalation_handler
from whatsapp.whatsapp_sender import WhatsAppSender, extract_incoming_text_message
import time

app = FastAPI()

VERIFY_TOKEN = config.WHATSAPP_VERIFY_TOKEN

rag_chain = chain_builder.build_rag_chain()
response_cache = get_response_cache()
query_logger = get_query_logger()
escalation_handler = get_escalation_handler()

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

        
        start_time = time.time()
        source_urls = [] # Default to empty for cache hits right now, could be enhanced later
        cache_hit = False

        escalation_response = escalation_handler.check_escalation(message_body)
        
        if escalation_response:
            print("Escalation triggered — skipping LLM call")
            clean_answer = escalation_response
        else:
            cached_answer = response_cache.get(message_body)
            if cached_answer is not None:
                print("Cache hit — skipping LLM call")
                clean_answer = cached_answer
                cache_hit = True
            else:
                response = await rag_chain.ainvoke({
                    "question": message_body,
                    "chat_history": ""
                })
                clean_answer = clean_llm_response(response)
                response_cache.put(message_body, clean_answer)

        response_time_ms = (time.time() - start_time) * 1000

        # Log the interaction
        query_logger.log_query(
            question=message_body,
            answer=clean_answer,
            source_urls=source_urls,
            response_time_ms=response_time_ms,
            cache_hit=cache_hit,
            platform="whatsapp"
        )

        await whatsapp_sender.send_text(sender_phone, clean_answer)

        return {"status": "success"}
    except Exception as e:
        print(f"Error handling webhook payload: {e}")
        return {"status": "ignored"}


class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def call_chat_loop(payload: ChatRequest):
    try:
        print(f"Processing incoming text: {payload.message}")

        start_time = time.time()
        source_urls = []
        cache_hit = False

        escalation_response = escalation_handler.check_escalation(payload.message)
        
        if escalation_response:
            print("Escalation triggered — skipping LLM call")
            clean_answer = escalation_response
        else:
            cached_answer = response_cache.get(payload.message)
            if cached_answer is not None:
                print("Cache hit — skipping LLM call")
                clean_answer = cached_answer
                cache_hit = True
            else:
                response = await rag_chain.ainvoke({
                    "question": payload.message,
                    "chat_history": ""
                })
                clean_answer = clean_llm_response(response)
                response_cache.put(payload.message, clean_answer)

        response_time_ms = (time.time() - start_time) * 1000
        
        # Log the interaction
        query_logger.log_query(
            question=payload.message,
            answer=clean_answer,
            source_urls=source_urls,
            response_time_ms=response_time_ms,
            cache_hit=cache_hit,
            platform="chat_api"
        )

        return {"status": "success", "answer": clean_answer, "cached": cache_hit}
    except Exception as e:
        print(f"Error handling chat request: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/analytics")
async def get_analytics():
    """
    Description: Returns basic analytics stats and recent queries.
    """
    stats = query_logger.get_stats()
    recent = query_logger.get_recent_queries(limit=10)
    return {
        "status": "success",
        "stats": stats,
        "recent_queries": recent
    }