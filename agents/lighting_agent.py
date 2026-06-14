"""
Lighting Agent
Receives the scene spec + composition summary and outputs:
  - bpy Python commands to set up all lights and world settings
  - a JSON summary of the lighting setup (fed to QA Agent)
"""

import anthropic
import json
import re


SYSTEM_PROMPT = """You are a professional Blender lighting artist. You receive a scene spec and
a composition summary.

Respond in this exact format — two tagged sections, nothing outside them:

<summary>
{
  "lights_added": [
    {"name": "<light name>", "type": "<POINT|SUN|SPOT|AREA>", "role": "<key|fill|rim|ambient>"}
  ],
  "world_background": "<description>",
  "mood_achieved": "<description>"
}
</summary>
<commands>
# --- COMMAND 1: render engine ---
bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
bpy.context.scene.eevee.taa_render_samples = 64
result = "done"
===
# --- COMMAND 2: world background ---
world = bpy.context.scene.world
world.use_nodes = True
bg = world.node_tree.nodes['Background']
bg.inputs['Color'].default_value = (0.01, 0.01, 0.02, 1.0)
bg.inputs['Strength'].default_value = 0.5
result = "done"
===
# --- COMMAND N: key light ---
bpy.ops.object.light_add(type='SPOT', location=(3, -3, 6))
light = bpy.context.object
light.name = 'KeyLight'
light.data.energy = 800
light.data.color = (1.0, 0.9, 0.7)
light.data.shadow_soft_size = 0.5
result = "done"
</commands>

Rules:
- Set render engine to 'BLENDER_EEVEE_NEXT' (Blender 5) in the first command.
- Convert hex colors to linear RGB floats manually (e.g. #ff8800 → (1.0, 0.53, 0.0)).
- Use bpy.ops.object.light_add(type=..., location=(...)) for each light.
- Configure world background via bpy.context.scene.world.node_tree nodes.
- If you use math.radians or any math function, add "import math" at the top of THAT command block.
- Separate every command block with === on its own line.
- Do NOT wrap code in JSON strings — write raw Python directly inside <commands>.
- Do NOT use markdown fences anywhere.
- Each command block must end with: result = "done"
"""


def _parse(raw: str) -> dict:
    summary_match = re.search(r"<summary>(.*?)</summary>", raw, re.DOTALL)
    commands_match = re.search(r"<commands>(.*?)</commands>", raw, re.DOTALL)

    if not summary_match or not commands_match:
        raise ValueError(f"Missing <summary> or <commands> tags in response:\n{raw[:500]}")

    summary = json.loads(summary_match.group(1).strip())
    commands_raw = commands_match.group(1).strip()
    commands = [c.strip() for c in commands_raw.split("===") if c.strip()]

    return {"summary": summary, "bpy_commands": commands}


def run(scene_spec: dict, composition_summary: dict, client: anthropic.Anthropic) -> dict:
    """
    Generate bpy commands to light the scene.
    Returns dict with bpy_commands (list[str]) and summary (dict).
    """
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=6000,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Scene spec:\n{json.dumps(scene_spec, indent=2)}\n\n"
                    f"Composition summary:\n{json.dumps(composition_summary, indent=2)}\n\n"
                    "Generate the <summary> and <commands> to light this scene. "
                    "Match the mood, time of day, and key/fill colors from the spec."
                ),
            }
        ],
    )

    raw = message.content[0].text.strip()
    return _parse(raw)
