# PromptPilot - AI Prompt Optimizer

> Rewrite rough prompts into structured, chain-of-thought enhanced prompts using GPT-3.5-turbo.

## What It Does

PromptPilot takes your messy, vague, or incomplete AI prompts and transforms them into:
- **Structured prompts** with clear role, context, task, and output format sections
- **Chain-of-thought enhanced** versions that guide the model to reason step-by-step
- **Few-shot ready** prompts with example placeholders
- **Token-efficient** rewrites that pack more meaning into fewer words

## Demo

```
Input:  "explain machine learning"

Output: "You are an expert ML educator. Your task is to explain machine learning
         to a beginner. Think step-by-step:
         1. Start with a simple real-world analogy
         2. Define the core concept in one sentence
         3. Describe the three main types (supervised, unsupervised, reinforcement)
         4. Give one concrete example for each type
         Format your response with clear headers and bullet points."
```

## Features

- **Flask web UI** - simple, clean browser interface
- **REST API** - `POST /optimize` endpoint for programmatic use
- **Multiple optimization modes**: structured, chain-of-thought, concise, few-shot
- **Before/after diff view** - see exactly what changed
- **Copy-to-clipboard** - one click to grab your optimized prompt
- **Prompt history** - local session history of all optimizations

## Tech Stack

- **Backend**: Python, Flask
- **AI**: OpenAI GPT-3.5-turbo
- **Frontend**: HTML/CSS/JS (vanilla, no framework)
- **Deployment**: Works locally or on any cloud (Render, Railway, Heroku)

## Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key

### Installation

```bash
git clone https://github.com/muhibwqr/promptpilot-prompt-optimizer.git
cd promptpilot-prompt-optimizer
pip install -r requirements.txt
```

### Configuration

```bash
export OPENAI_API_KEY=your_api_key_here
```

Or create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here
```

### Run

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

## API Usage

```bash
curl -X POST http://localhost:5000/optimize \
  -H "Content-Type: application/json" \
  -d '{"prompt": "explain machine learning", "mode": "chain-of-thought"}'
```

**Response:**
```json
{
  "original": "explain machine learning",
  "optimized": "...",
  "mode": "chain-of-thought",
  "tokens_saved": 0,
  "improvement_score": 8.5
}
```

## Optimization Modes

| Mode | Description |
|------|-------------|
| `structured` | Adds role, context, task, and output format sections |
| `chain-of-thought` | Adds step-by-step reasoning instructions |
| `concise` | Removes fluff, maximizes token efficiency |
| `few-shot` | Adds example placeholders for few-shot learning |
| `auto` | GPT picks the best mode for your prompt |

## Project Structure

```
promotpilot-prompt-optimizer/
├── app.py              # Flask app + optimization logic
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html      # Web UI
└── README.md
```

## Built With

- [OpenAI API](https://platform.openai.com/) - GPT-3.5-turbo for prompt rewriting
- [Flask](https://flask.palletsprojects.com/) - Lightweight Python web framework
- [python-dotenv](https://pypi.org/project/python-dotenv/) - Environment variable management

## Contributing

Pull requests welcome! Open an issue first to discuss what you'd like to change.

## License

MIT License - see [LICENSE](LICENSE) for details.

---
*Built for the Prompt-a-Thon Hackathon, March 2026*
