"""
brain.py – HiveRift Intelligence Brain (Groq Edition)
Fixed: message validation to prevent 400 errors from Groq API
"""

import os
import json
import requests
from collections import deque

# ── Config ─────────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant"

MAX_NEW_TOKENS = 512
MEMORY_TURNS   = 5

KB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "Knowledge Base", "hiverift_source_of_truth.md"
)

# ── Load full KB once at startup ───────────────────────────────────────────────
def _load_kb() -> str:
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "No knowledge base found."

KNOWLEDGE_BASE = _load_kb()

# ── Out-of-scope guardrail ─────────────────────────────────────────────────────
OUT_OF_SCOPE = [
    "politics","election","government","president","prime minister",
    "modi","trump","biden","congress","parliament",
    "competitor","versus","vs ","alternative to","better than",
    "stock","bitcoin","crypto","nft",
    "recipe","cook","movie","sport","cricket","football","weather",
    "news","joke","poem","story","essay",
]

# ── Sales phase detector ───────────────────────────────────────────────────────
PHASE_SIGNALS = {
    "Phase 1 - Lead Capture":   ["hello","hi","hey","interested","tell me about","what do you do"],
    "Phase 2 - Qualification":  ["budget","how much","afford","cost","price","can you help","do you offer"],
    "Phase 3 - Needs Analysis": ["need","requirement","want","looking for","my project","we need","use case","details","technical"],
    "Phase 4 - Proposal":       ["proposal","quote","send me","can you send","document","breakdown"],
    "Phase 5 - Presentation":   ["explain","walk me through","how does it work","can you show","demo"],
    "Phase 6 - Negotiation":    ["discount","negotiate","reduce","lower price","can you do better","deal","offer"],
    "Phase 7 - Closure":        ["agree","proceed","let's go","sign","payment","advance","start the project","finalise"],
}

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the HiveRift Intelligence Brain, a specialised AI sales assistant for HiveRift, a digital services agency in India.
Your personality: professional, warm, concise, and consultative.
COMPLETE HIVERIFT KNOWLEDGE BASE:
{knowledge_base}
RULES (never violate):
- Answer ONLY using information from the Knowledge Base above. Do not invent any facts, prices, timelines, or names.
- If the answer is not in the Knowledge Base, say: "I don't have that detail, please contact us at hello@hiverift.com or call +91-11-4567-8900."
- Prices are always in INR. Always append: "All prices are indicative and subject to final scope confirmation."
- Maximum discount you may mention: 15%.
- Never discuss competitors or topics outside HiveRift services.
- Keep answers to 3-5 sentences. Be conversational, not robotic.
- End each reply with a gentle next-step nudge.
Current Sales Phase: {phase}"""


# ── Memory ─────────────────────────────────────────────────────────────────────
class ConversationMemory:
    def __init__(self, k: int = MEMORY_TURNS):
        self.k = k
        self._buf: deque = deque(maxlen=k * 2)

    def add(self, role: str, content: str):
        # Only add if content is a non-empty string
        if content and isinstance(content, str) and content.strip():
            self._buf.append({"role": role, "content": content.strip()})

    def messages(self) -> list:
        return list(self._buf)

    def clear(self):
        self._buf.clear()

_memory = ConversationMemory()


# ── Helpers ────────────────────────────────────────────────────────────────────
def _is_out_of_scope(q: str) -> bool:
    return any(kw in q.lower() for kw in OUT_OF_SCOPE)

def _detect_phase(q: str) -> str:
    ql = q.lower()
    for phase, signals in PHASE_SIGNALS.items():
        if any(s in ql for s in signals):
            return phase
    return "Phase 3 - Needs Analysis"

def _headers() -> dict:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. "
            "Go to https://console.groq.com, create a key, "
            "then add it to your HF Space secrets as GROQ_API_KEY."
        )
    return {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

def _build_messages(user_message: str) -> list:
    """Build a clean, validated message list for the Groq API."""
    phase = _detect_phase(user_message)
    system = SYSTEM_PROMPT.format(
        knowledge_base=KNOWLEDGE_BASE,
        phase=phase
    )

    messages = []

    # 1. System message
    messages.append({
        "role": "system",
        "content": system
    })

    # 2. Conversation history — validate every entry
    for msg in _memory.messages():
        if (
            isinstance(msg, dict)
            and msg.get("role") in ("user", "assistant")
            and isinstance(msg.get("content"), str)
            and msg["content"].strip()
        ):
            messages.append({
                "role": msg["role"],
                "content": msg["content"].strip()
            })

    # 3. Current user message
    messages.append({
        "role": "user",
        "content": user_message.strip()
    })

    return messages


def _handle_http_error(resp) -> str:
    """Return a clean error message based on status code."""
    if resp.status_code == 400:
        try:
            detail = resp.json().get("error", {}).get("message", "Bad request")
        except Exception:
            detail = resp.text[:200]
        return f"Request error: {detail}"
    if resp.status_code == 401:
        return "Invalid GROQ_API_KEY. Please check your Space secret."
    if resp.status_code == 429:
        return "Rate limit reached. Please try again in a moment."
    if resp.status_code == 503:
        return "Groq service temporarily unavailable. Please try again shortly."
    return f"Groq API error ({resp.status_code}). Please contact hello@hiverift.com."


# ── Non-streaming query ────────────────────────────────────────────────────────
def query(user_message: str) -> str:
    if not user_message or not user_message.strip():
        return "Please enter a message."

    if _is_out_of_scope(user_message):
        return (
            "I'm specialised for HiveRift topics only.\n\n"
            "Please reach out to **hello@hiverift.com** for anything else."
        )

    messages = _build_messages(user_message)
    _memory.add("user", user_message)

    try:
        resp = requests.post(
            GROQ_API_URL,
            headers=_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": MAX_NEW_TOKENS,
                "temperature": 0.35,
                "top_p": 0.9,
                "stream": False,
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()

    except EnvironmentError as e:
        return f"Configuration error: {e}"
    except requests.exceptions.HTTPError:
        return _handle_http_error(resp)
    except requests.exceptions.Timeout:
        return "Request timed out. Please try again."
    except Exception as e:
        return f"Unexpected error: {e}"

    _memory.add("assistant", answer)
    return answer


# ── Streaming query ────────────────────────────────────────────────────────────
def query_stream(user_message: str):
    """Yields text tokens one by one for SSE streaming."""
    if not user_message or not user_message.strip():
        yield "Please enter a message."
        return

    if _is_out_of_scope(user_message):
        yield "I'm specialised for HiveRift topics only. Please contact **hello@hiverift.com**."
        return

    messages = _build_messages(user_message)
    _memory.add("user", user_message)

    try:
        with requests.post(
            GROQ_API_URL,
            headers=_headers(),
            json={
                "model": GROQ_MODEL,
                "messages": messages,
                "max_tokens": MAX_NEW_TOKENS,
                "temperature": 0.35,
                "top_p": 0.9,
                "stream": True,
            },
            stream=True,
            timeout=30,
        ) as resp:
            if not resp.ok:
                yield _handle_http_error(resp)
                return

            full = []
            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if raw == "[DONE]":
                    break
                try:
                    delta = json.loads(raw)["choices"][0]["delta"]
                    token = delta.get("content", "")
                    if token:
                        full.append(token)
                        yield token
                except Exception:
                    continue

            _memory.add("assistant", "".join(full))

    except EnvironmentError as e:
        yield f"Configuration error: {e}"
    except requests.exceptions.Timeout:
        yield "Request timed out. Please try again."
    except Exception as e:
        yield f"Error: {e}"


def clear_memory():
    _memory.clear()
