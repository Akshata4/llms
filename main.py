import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")

if not API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set. Put it in .env or your environment.")

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

def pick_free_model() -> str:
    """Return the first free model available."""
    with httpx.Client(timeout=30) as client:
        r = client.get(f"{BASE_URL}/models", headers=HEADERS)
        r.raise_for_status()
        data = r.json()

    for m in data.get("data", []):
        pricing = m.get("pricing") or {}
        prompt_price = float(pricing.get("prompt", 0) or 0)
        completion_price = float(pricing.get("completion", 0) or 0)
        if prompt_price == 0 and completion_price == 0:
            return m["id"]

    # Fallback if no free model is marked
    return "gryphe/mythomist-7b"

def chat(messages, model=None, temperature=0.7):
    if model is None:
        model = pick_free_model()

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    with httpx.Client(timeout=60) as client:
        r = client.post(f"{BASE_URL}/chat/completions", headers=HEADERS, content=json.dumps(payload))
        r.raise_for_status()
        data = r.json()

    return {
        "model": model,
        "content": data["choices"][0]["message"]["content"]
    }

if __name__ == "__main__":
    msgs = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain transformers in one short paragraph."},
    ]
    result = chat(msgs)
    print(f"Model: {result['model']}\n")
    print(result["content"])
