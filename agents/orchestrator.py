"""
Orchestrator Agent
Validates the user prompt and transforms it into a structured JSON scene spec
that all downstream agents consume.
"""

import anthropic
import json


SYSTEM_PROMPT = """You are a 3D scene director. Given a natural-language prompt,
output a structured JSON scene specification that downstream agents will use to build
the scene in Blender.

Output ONLY valid JSON — no markdown fences, no commentary. Structure:
{
  "scene_description": "<one-sentence summary>",
  "mood": "<e.g. dark and moody, warm and cozy>",
  "time_of_day": "<day | golden hour | night | dusk | dawn>",
  "style": "<e.g. cyberpunk, fantasy, realism, cartoon>",
  "objects": [
    {"name": "<object name>", "role": "<hero | background | prop>", "placement": "<left | centre | right | foreground | background>", "count": 1}
  ],
  "camera": {
    "angle": "<eye level | low angle | high angle | birds eye>",
    "framing": "<wide | medium | close-up>",
    "focal_length_mm": 35
  },
  "lighting": {
    "style": "<e.g. three-point, moonlight, neon, candle>",
    "key_color_hex": "#ffffff",
    "key_intensity": 5.0,
    "fill_color_hex": "#ffffff",
    "fill_intensity": 1.0,
    "use_hdri": false,
    "ambient_strength": 0.1
  },
  "render": {
    "engine": "EEVEE",
    "samples": 64,
    "resolution_x": 1280,
    "resolution_y": 720
  }
}

Be specific and faithful to the prompt. Choose values that will produce a visually compelling scene.
"""


def run(prompt: str, client: anthropic.Anthropic, feedback: str = "") -> dict:
    """
    Transform a user prompt into a structured scene spec JSON.
    If feedback is provided (from QA retry), incorporate it.
    """
    user_message = f"Create a scene spec for: {prompt}"
    if feedback:
        user_message += f"\n\nQA feedback from previous attempt (incorporate these fixes):\n{feedback}"

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = message.content[0].text.strip()
    try:
        spec = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        spec = json.loads(raw[start:end])

    return spec
