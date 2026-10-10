
from flask import Flask, request, jsonify
from orchestrator import analyze_source

app = Flask(__name__)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.get_json(silent=True) or {}
    code = data.get("code", "")

    if not isinstance(code, str) or not code.strip():
        return jsonify({
            "status": "error",
            "error": "Please enter some Python code to analyze."
        }), 400

    try:
        report = analyze_source(code)
        return jsonify(report), 200
    except Exception:
        app.logger.exception("Code analysis failed")
        return jsonify({
            "status": "error",
            "error": "Analysis failed because of an internal server error."
        }), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)