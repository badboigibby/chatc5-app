# app.py
# Hugging Face Space backend for ChatC5 model

from flask import Flask, request, jsonify
from llama_cpp import Llama
import os

# ----------------------------
# Config
# ----------------------------
MODEL_REPO = "badboigibby/chatc5-7.12B"  # Hugging Face model repo
MODEL_FILE = "chatc5.gguf"

SYSTEM_PROMPT = (
    "You are ChatC5 created by OAG. "
    "Answer helpfully and provide complete code when asked. "
    "Only output full HTML documents if the user explicitly requests HTML."
)

# ----------------------------
# Load model
# ----------------------------
llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_ctx=2048,
    n_threads=2,    # keep lower for Spaces
    n_batch=512
)

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({
        "message": "ChatC5 backend running. Use POST /chat with JSON {message: \"...\"}.",
        "frontend": "https://badboigibby.github.io/chatc5/"
    })

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    output = llm.create_chat_completion(
        messages=messages,
        stream=False,
        max_tokens=1024,
        temperature=0.4
    )

    reply = output["choices"][0]["message"]["content"].strip()
    return jsonify({"reply": reply})


# ----------------------------
# Entrypoint
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(host="0.0.0.0", port=port)
