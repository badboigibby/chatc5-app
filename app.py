import os
from flask import Flask, request, jsonify, render_template
from llama_cpp import Llama

# ----------------------------
# Config
# ----------------------------
MODEL_REPO = "badboigibby/chatc5-7.12B"  # Hugging Face repo
MODEL_FILE = "chatc5.gguf"               # Local model file

SYSTEM_PROMPT = (
    "You are ChatC5 created by OAG. "
    "Answer helpfully and provide complete code when asked. "
    "Only output full HTML documents if the user explicitly requests HTML."
)

# ----------------------------
# Load Model
# ----------------------------
llm = Llama.from_pretrained(
    repo_id=MODEL_REPO,
    filename=MODEL_FILE,
    n_ctx=2048,
    n_threads=2,
    n_batch=512
)

# ----------------------------
# Flask App
# ----------------------------
app = Flask(__name__, template_folder="templates")

# Serve the Sky Chat UI
@app.route("/")
def home():
    return render_template("chat.html")

# Chat API endpoint
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "No message provided"}), 400

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
# Run App
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT"))
    app.run(host="0.0.0.0", port=port)
