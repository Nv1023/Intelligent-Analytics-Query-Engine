from pathlib import Path

from flask import Flask, jsonify, render_template, request

from analytics_engine import QueryEngine

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

BASE_DIR = Path(__file__).resolve().parent
engine = QueryEngine(str(BASE_DIR / "dataset"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask():
    payload = request.get_json(silent=True) or {}
    question = (payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Please enter a question."}), 400

    response = engine.chat_answer(question)
    return jsonify(response)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
