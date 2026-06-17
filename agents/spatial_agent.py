"""
Spatial Agent
Runs after composition, before lighting.
Looks at a viewport screenshot + scene JSON and fixes spatial issues
(floating objects, wrong scale, out-of-frame props, clipping geometry)
before the lighting agent runs.

This directly addresses the known limitation that LLMs place objects
"approximately" and require several rounds of correction.
"""

import anthropic
import json
import re


SYSTEM_PROMPT = """You are a 3D scene spatial reviewer. You receive:
1. A viewport screenshot of the scene as it currently looks.
2. A JSON list of every object (name, type, location, scale).
3. The original scene spec that guided the build.

Your job is to spot and fix spatial problems BEFORE lighting runs.

Look for:
- FLOATING objects — props, rocks, pillars, or any ground-level object whose Z location
  puts it visibly above the floor (floor is usually Z=0 or the ground plane scale).
- WRONG SCALE — an object that is clearly too large or too small relative to its neighbours
  (e.g. a lamp post the same height as a skyscraper, or a rock the size of a building).
- OUT OF FRAME — objects placed so far from the camera that they are invisible and wasted.
- CLIPPING — two objects occupying nearly the same location and z-fighting.
- HERO CHARACTER blocked — if an armature/character exists, make sure nothing large sits
  directly in front of it.

DO NOT touch:
- Lights or world settings (lighting agent handles those)
- Camera (separate concern)
- Materials or node trees
- The scale of any ARMATURE object (Mixamo character scale is intentional)

Respond in this exact format — nothing outside the two tags:

<issues>
["short description of issue 1", "short description of issue 2"]
</issues>
<fixes>
# fix 1 — brief comment
obj = bpy.data.objects.get('ObjectName')
if obj:
    obj.location.z = 0.0
result = "done"
===
# fix 2
obj = bpy.data.objects.get('OtherObject')
if obj:
    obj.scale = (1.0, 1.0, 1.0)
result = "done"
</fixes>

If no issues are found, output:
<issues>[]</issues>
<fixes></fixes>

Rules:
- Only fix what you can clearly see is wrong from the screenshot or the JSON data.
- Each fix block must end with: result = "done"
- Separate fix blocks with === on its own line.
- No markdown fences. No text outside the two tags.
- Keep fixes minimal — move or scale only, never delete or recreate objects.
"""


def run(
    scene_spec: dict,
    scene_snapshot: str,
    client: anthropic.Anthropic,
    screenshot_b64: str | None = None,
) -> dict:
    """
    Review the scene for spatial problems and return correction commands.
    Returns dict with: issues (list[str]), fixes (list[str])
    """
    text_content = (
        f"Scene spec:\n{json.dumps(scene_spec, indent=2)}\n\n"
        f"Scene objects (JSON):\n{scene_snapshot}\n\n"
        "Review for spatial issues and output <issues> and <fixes>."
    )

    if screenshot_b64:
        user_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": screenshot_b64,
                },
            },
            {"type": "text", "text": text_content},
        ]
    else:
        user_content = text_content

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )

    raw = message.content[0].text.strip()

    issues_match = re.search(r"<issues>(.*?)</issues>", raw, re.DOTALL)
    fixes_match = re.search(r"<fixes>(.*?)</fixes>", raw, re.DOTALL)

    issues_raw = issues_match.group(1).strip() if issues_match else "[]"
    try:
        issues = json.loads(issues_raw)
    except json.JSONDecodeError:
        issues = [issues_raw] if issues_raw and issues_raw != "[]" else []

    fixes_raw = fixes_match.group(1).strip() if fixes_match else ""
    fixes = [f.strip() for f in fixes_raw.split("===") if f.strip()] if fixes_raw else []

    return {"issues": issues, "fixes": fixes}
