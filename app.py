import os
import json
from flask import Flask, request, jsonify, render_template_string
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Optimization system prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = {
    "structured": """You are an expert prompt engineer. Your job is to rewrite the user's rough prompt 
into a well-structured prompt with these sections:
- Role: Define who the AI should be
- Context: Provide relevant background
- Task: State the specific task clearly
- Constraints: Any limitations or rules
- Output Format: Specify how the answer should be formatted

Return ONLY the rewritten prompt, no explanation.""",

    "chain-of-thought": """You are an expert prompt engineer specializing in chain-of-thought prompting.
Rewrite the user's prompt to include explicit step-by-step reasoning instructions.
The rewritten prompt should:
1. Assign an expert role to the AI
2. Ask the AI to think step-by-step before answering
3. Break the problem into logical sub-tasks
4. Ask for a final summary after the reasoning

Return ONLY the rewritten prompt, no explanation.""",

    "concise": """You are an expert prompt engineer focused on token efficiency.
Rewrite the user's prompt to be as concise as possible while preserving all intent.
Remove filler words, redundancies, and vague language.
Use precise, action-oriented vocabulary.

Return ONLY the rewritten prompt, no explanation.""",

    "few-shot": """You are an expert prompt engineer specializing in few-shot prompting.
Rewrite the user's prompt to include:
1. A clear task description
2. 2-3 labeled example pairs (Example Input / Example Output) with [PLACEHOLDER] content
3. The actual task at the end following the same format

Return ONLY the rewritten prompt with placeholders, no explanation.""",

    "auto": """You are an expert prompt engineer. Analyze the user's rough prompt and choose 
the single best optimization strategy from: structured, chain-of-thought, concise, or few-shot.
Then apply that strategy to rewrite the prompt.

First line of your response: Strategy: <chosen_strategy>
Then an empty line.
Then the rewritten prompt only."""
}

# ---------------------------------------------------------------------------
# HTML template (inline for single-file simplicity)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PromptPilot - AI Prompt Optimizer</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0f0f1a; color: #e2e8f0; min-height: 100vh; }
        header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                 padding: 20px 40px; display: flex; align-items: center; gap: 12px; }
        header h1 { font-size: 1.8rem; font-weight: 700; color: white; }
        header span { font-size: 0.9rem; color: rgba(255,255,255,0.8); }
        .container { max-width: 900px; margin: 40px auto; padding: 0 20px; }
        .card { background: #1a1a2e; border: 1px solid #2d2d44; border-radius: 12px;
                padding: 24px; margin-bottom: 24px; }
        label { display: block; font-size: 0.85rem; font-weight: 600;
                color: #a0aec0; margin-bottom: 8px; text-transform: uppercase;
                letter-spacing: 0.05em; }
        textarea { width: 100%; background: #0f0f1a; border: 1px solid #2d2d44;
                   border-radius: 8px; padding: 14px; color: #e2e8f0;
                   font-size: 0.95rem; resize: vertical; min-height: 120px;
                   font-family: inherit; transition: border-color 0.2s; }
        textarea:focus { outline: none; border-color: #667eea; }
        .mode-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 8px;
                     margin-top: 8px; }
        .mode-btn { padding: 8px 4px; border: 1px solid #2d2d44; border-radius: 8px;
                    background: #0f0f1a; color: #a0aec0; cursor: pointer;
                    font-size: 0.78rem; text-align: center; transition: all 0.2s; }
        .mode-btn:hover, .mode-btn.active { border-color: #667eea; color: #667eea;
                                            background: rgba(102,126,234,0.1); }
        .optimize-btn { width: 100%; padding: 14px; background: linear-gradient(135deg, #667eea, #764ba2);
                        border: none; border-radius: 8px; color: white; font-size: 1rem;
                        font-weight: 600; cursor: pointer; margin-top: 16px;
                        transition: opacity 0.2s; }
        .optimize-btn:hover { opacity: 0.9; }
        .optimize-btn:disabled { opacity: 0.5; cursor: not-allowed; }
        .result-header { display: flex; justify-content: space-between; align-items: center;
                         margin-bottom: 12px; }
        .copy-btn { padding: 6px 14px; background: transparent; border: 1px solid #667eea;
                    border-radius: 6px; color: #667eea; font-size: 0.8rem; cursor: pointer;
                    transition: all 0.2s; }
        .copy-btn:hover { background: rgba(102,126,234,0.1); }
        .score-badge { display: inline-flex; align-items: center; gap: 6px;
                       background: rgba(102,126,234,0.15); border: 1px solid rgba(102,126,234,0.3);
                       border-radius: 20px; padding: 4px 12px; font-size: 0.82rem; color: #a78bfa; }
        .history-item { padding: 12px; border: 1px solid #2d2d44; border-radius: 8px;
                        margin-bottom: 8px; cursor: pointer; transition: border-color 0.2s; }
        .history-item:hover { border-color: #667eea; }
        .history-item .meta { font-size: 0.75rem; color: #4a5568; margin-top: 4px; }
        .spinner { display: inline-block; width: 16px; height: 16px;
                   border: 2px solid rgba(255,255,255,0.3); border-top-color: white;
                   border-radius: 50%; animation: spin 0.8s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        #result-section { display: none; }
    </style>
</head>
<body>
    <header>
        <h1>PromptPilot</h1>
        <span>AI Prompt Optimizer</span>
    </header>
    <div class="container">
        <div class="card">
            <label>Your Rough Prompt</label>
            <textarea id="input-prompt" placeholder="e.g. explain machine learning, write me a cover letter, summarize this article..."></textarea>
            <label style="margin-top:16px;">Optimization Mode</label>
            <div class="mode-grid">
                <div class="mode-btn active" data-mode="auto" onclick="selectMode(this)">Auto</div>
                <div class="mode-btn" data-mode="structured" onclick="selectMode(this)">Structured</div>
                <div class="mode-btn" data-mode="chain-of-thought" onclick="selectMode(this)">Chain-of-Thought</div>
                <div class="mode-btn" data-mode="concise" onclick="selectMode(this)">Concise</div>
                <div class="mode-btn" data-mode="few-shot" onclick="selectMode(this)">Few-Shot</div>
            </div>
            <button class="optimize-btn" onclick="optimize()" id="opt-btn">Optimize Prompt</button>
        </div>

        <div class="card" id="result-section">
            <div class="result-header">
                <label style="margin:0;">Optimized Prompt <span id="mode-label" class="score-badge"></span></label>
                <button class="copy-btn" onclick="copyResult()">Copy</button>
            </div>
            <textarea id="output-prompt" style="min-height:160px;"></textarea>
        </div>

        <div class="card" id="history-card" style="display:none;">
            <label>Session History</label>
            <div id="history-list"></div>
        </div>
    </div>

    <script>
        let selectedMode = 'auto';
        let history = [];

        function selectMode(el) {
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            el.classList.add('active');
            selectedMode = el.dataset.mode;
        }

        async function optimize() {
            const prompt = document.getElementById('input-prompt').value.trim();
            if (!prompt) return alert('Please enter a prompt first.');

            const btn = document.getElementById('opt-btn');
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Optimizing...';

            try {
                const res = await fetch('/optimize', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, mode: selectedMode})
                });
                const data = await res.json();
                if (data.error) throw new Error(data.error);

                document.getElementById('output-prompt').value = data.optimized;
                document.getElementById('mode-label').textContent = data.mode;
                document.getElementById('result-section').style.display = 'block';

                history.unshift({original: prompt, optimized: data.optimized, mode: data.mode});
                renderHistory();
            } catch(e) {
                alert('Error: ' + e.message);
            } finally {
                btn.disabled = false;
                btn.textContent = 'Optimize Prompt';
            }
        }

        function copyResult() {
            const text = document.getElementById('output-prompt').value;
            navigator.clipboard.writeText(text).then(() => {
                const btn = document.querySelector('.copy-btn');
                btn.textContent = 'Copied!';
                setTimeout(() => btn.textContent = 'Copy', 1500);
            });
        }

        function renderHistory() {
            if (history.length === 0) return;
            document.getElementById('history-card').style.display = 'block';
            const list = document.getElementById('history-list');
            list.innerHTML = history.map((h, i) => `
                <div class="history-item" onclick="loadHistory(${i})">
                    <div>${h.original.substring(0, 80)}${h.original.length > 80 ? '...' : ''}</div>
                    <div class="meta">Mode: ${h.mode}</div>
                </div>
            `).join('');
        }

        function loadHistory(i) {
            document.getElementById('input-prompt').value = history[i].original;
            document.getElementById('output-prompt').value = history[i].optimized;
            document.getElementById('mode-label').textContent = history[i].mode;
            document.getElementById('result-section').style.display = 'block';
        }
    </script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/optimize", methods=["POST"])
def optimize():
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "Missing 'prompt' in request body"}), 400

    user_prompt = data["prompt"].strip()
    mode = data.get("mode", "auto").lower()

    if mode not in SYSTEM_PROMPTS:
        mode = "auto"

    system = SYSTEM_PROMPTS[mode]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=800
        )

        optimized_text = response.choices[0].message.content.strip()
        detected_mode = mode

        # Extract strategy line for auto mode
        if mode == "auto" and optimized_text.startswith("Strategy:"):
            lines = optimized_text.split("\n", 2)
            if len(lines) >= 1:
                detected_mode = lines[0].replace("Strategy:", "").strip().lower()
            optimized_text = "\n".join(lines[2:]).strip() if len(lines) > 2 else "\n".join(lines[1:]).strip()

        return jsonify({
            "original": user_prompt,
            "optimized": optimized_text,
            "mode": detected_mode,
            "model": "gpt-3.5-turbo"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "promptpilot"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
