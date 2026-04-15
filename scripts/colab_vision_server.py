#!/usr/bin/env python3
"""ORION Vision Server — Colab-ready Qwen2.5-VL inference server.

Copy this script into a Google Colab notebook cell-by-cell. Each section
marked with ``# ── CELL N`` should be placed in a separate Colab cell.

Prerequisites
-------------
- Google Colab with GPU runtime (T4 or better).
- Free ngrok account for tunnel auth token.

Usage
-----
1. Upload this file to Colab or copy cells manually.
2. Run Cell 1–4 in order.
3. Copy the printed ``VISION_API_URL`` to your local ``.env``.
"""

# ── CELL 1 — Install Dependencies ────────────────────────────────────
# !pip install -q transformers accelerate torch torchvision pyngrok flask pillow

# ── CELL 2 — Load Qwen2.5-VL Model ──────────────────────────────────
"""
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch

MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

print(f"Loading {MODEL_NAME}...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
model.eval()
print(f"Model loaded on {model.device}")
"""

# ── CELL 3 — Flask API Server ────────────────────────────────────────
"""
from flask import Flask, request, jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)


@app.route("/analyze", methods=["POST"])
def analyze():
    '''Analyze a screenshot with Qwen2.5-VL.

    Expects JSON body:
        image_base64 (str): Base64-encoded PNG/JPEG image.
        prompt (str, optional): Custom analysis prompt.

    Returns JSON:
        result (str): Model's analysis text.
        model (str): Model identifier.
    '''
    data = request.json
    img_b64 = data["image_base64"]
    prompt = data.get(
        "prompt",
        "Describe this screen. List all clickable UI elements with their "
        "approximate pixel coordinates as bounding boxes [x1,y1,x2,y2].",
    )

    # Decode image
    img_bytes = base64.b64decode(img_b64)
    image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Build Qwen2.5-VL messages
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=512)

    output = processor.decode(output_ids[0], skip_special_tokens=True)
    return jsonify({"result": output, "model": MODEL_NAME})


@app.route("/health", methods=["GET"])
def health():
    '''Health check endpoint.'''
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "device": str(model.device),
    })
"""

# ── CELL 4 — Start with ngrok Tunnel ────────────────────────────────
"""
from pyngrok import ngrok
import threading
import os

# Set your ngrok auth token (free at https://ngrok.com)
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN", "YOUR_NGROK_AUTHTOKEN")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Create tunnel
public_url = ngrok.connect(5000).public_url
print(f"")
print(f"  ╔══════════════════════════════════════════════════╗")
print(f"  ║  ORION Vision Server Ready                      ║")
print(f"  ║                                                  ║")
print(f"  ║  Vision API URL: {public_url:<30} ║")
print(f"  ║                                                  ║")
print(f"  ║  Add to your local .env:                         ║")
print(f"  ║  VISION_API_URL={public_url:<30}  ║")
print(f"  ╚══════════════════════════════════════════════════╝")
print(f"")

# Start Flask in background thread
threading.Thread(
    target=lambda: app.run(host="0.0.0.0", port=5000, debug=False),
    daemon=True,
).start()
"""
