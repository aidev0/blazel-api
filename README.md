# Blazel

AI-powered LinkedIn post generator with personalized LoRA fine-tuning.

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│  Frontend   │────▶│   API       │────▶│  Inference      │
│  (Next.js)  │     │  (FastAPI)  │     │  (vLLM + LoRA)  │
└─────────────┘     └──────┬──────┘     └─────────────────┘
                          │
                          ▼
                   ┌─────────────┐
                   │  Trainer    │
                   │  (LoRA)     │
                   └─────────────┘
```

## Services

| Service | Description | URL |
|---------|-------------|-----|
| **blazel-web** | Next.js frontend | https://blazel-web-19d3cd34dc51.herokuapp.com |
| **blazel-api** | FastAPI backend | https://blazel-api-9d69c876e191.herokuapp.com |
| **blazel-inference** | vLLM with LoRA adapters | GCP VM (35.229.82.124:8001) |
| **blazel-trainer** | LoRA fine-tuning service | GCP VM (34.168.168.186:8002) |

## Features

- Generate LinkedIn posts from topic + context
- Stream responses via SSE
- Collect user feedback (edits, ratings)
- Train personalized LoRA adapters per customer
- Hot-swap adapters without model reload

## Tech Stack

- **Frontend**: Next.js, TypeScript, Tailwind CSS
- **Backend**: FastAPI, MongoDB, WorkOS Auth
- **ML**: vLLM, Llama 3.1 8B, PEFT/LoRA
- **Infrastructure**: Heroku, GCP (T4 GPU)
