from flask import Flask, request, jsonify
from flask_cors import CORS
from main import generate_answer, load_model

app = Flask(__name__)
CORS(app)
load_model()

@app.route("/chat", methods=["GET"])
def chat():
    query = request.args.get("text", "").strip()
    if not query:
        return jsonify({"error": "請提供 text 的參數"}), 400

    # 原始完整回應
    full_output = generate_answer(query)

    # ✅ 過濾掉 system / user，只保留 assistant 最後的回應內容
    if "assistant" in full_output:
        reply = full_output.split("assistant")[-1].strip()
    else:
        reply = full_output.strip()

    return jsonify({"response": reply})

if __name__ == "__main__":
    app.run(debug=True)
