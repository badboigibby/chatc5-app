import os
from flask import Flask, request, jsonify
from llama_cpp import Llama

MODEL_REPO = "badboigibby/chatc5-7.12B"
MODEL_FILE = "chatc5.gguf"

SYSTEM_PROMPT = (
    "You are ChatC5 created by OAG. "
    "Answer helpfully and provide complete code when asked. "
    "Only output full HTML documents if the user explicitly requests HTML."
)

llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_ctx=2048,
    n_threads=2,
    n_batch=512
)

app = Flask(__name__)

@app.route("/")
def home():
    return {"message": "ChatC5 backend running on Railway!"}

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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Railway injects PORT
    app.run(host="0.0.0.0", port=port)
