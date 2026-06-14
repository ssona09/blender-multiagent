"""
QA Agent
Reviews the scene by inspecting the live Blender scene graph (object names, types, locations,
light energies). Scores it 1–10 against the original prompt. If score < 7, returns
actionable feedback for the Orchestrator to retry with, plus immediate bpy fix commands.
"""

import anthropic
import json
import re


SYSTEM_PROMPT = """You are a 3D scene QA reviewer. You receive:
1. The original user prompt.
2. The structured scene spec that was used to build it.
3. A JSON snapshot of every object currently in Blender (name, type, location, extra data).

Score the scene 1–10 on how faithfully and attractively it represents the prompt.

Respond in this exact format — two tagged sections, nothing outside them:

<review>
{
  "score": <1-10 integer>,
  "verdict": "pass or fail",
  "strengths": ["<what is working well>"],
  "issues": ["<specific problem>"],
  "feedback_for_orchestrator": "<paragraph — what to do differently on retry>"
}
</review>
<fixes>
# only include if there are quick fixes to apply immediately
bpy.context.object.location = (0, 0, 0)
result = "done"
===
# another fix block (separate with ===)
</fixes>

If no immediate fixes are needed, output empty <fixes></fixes> tags.

Scoring guide:
- 9-10: Excellent match, professional look
- 7-8: Good match, minor issues (verdict = pass)
- 5-6: Partial match, missing key elements (verdict = fail)
- 1-4: Poor match, fundamental issues (verdict = fail)

verdict is "pass" if score >= 7, else "fail".
immediate_fixes are small bpy commands to apply right now without a full retry.
feedback_for_orchestrator is strategic guidance for the next full attempt.

IMPORTANT — fixes must only use safe bpy operations:
- Move/scale/rotate objects: obj.location, obj.scale, obj.rotation_euler
- Camera focal length: bpy.context.scene.camera.data.lens = 24  (NOT cam.focal_length)
- Add lights: bpy.ops.object.light_add(), light.data.energy, light.data.color
- World background: bpy.context.scene.world.node_tree nodes
- DO NOT touch materials, Specular, or node trees on mesh objects — Blender 5 removed many old nodes.
- DO NOT change the scale of any imported character/armature — it is handled by the importer.
If verdict is "pass", immediate_fixes can be empty.
"""

SCENE_SNAPSHOT_CODE = """
import json, math
import bpy

# Find hero character position (armature or mesh named Akai/Armature)
hero = None
for obj in bpy.context.scene.objects:
    if obj.type == 'ARMATURE' or 'akai' in obj.name.lower():
        hero = obj
        break
hero_loc = list(hero.location) if hero else [0, 0, 0]

snapshot = []
for obj in bpy.context.scene.objects:
    entry = {
        "name": obj.name,
        "type": obj.type,
        "location": [round(v, 3) for v in obj.location],
        "scale": [round(v, 3) for v in obj.scale],
    }
    if obj.type == "LIGHT":
        entry["light_type"] = obj.data.type
        entry["energy"] = round(obj.data.energy, 2)
        entry["color"] = [round(c, 3) for c in obj.data.color]
        # Distance from this light to the hero character
        dx = obj.location.x - hero_loc[0]
        dy = obj.location.y - hero_loc[1]
        dz = obj.location.z - hero_loc[2]
        dist = round(math.sqrt(dx*dx + dy*dy + dz*dz), 2)
        entry["dist_to_hero"] = dist
        entry["likely_hits_hero"] = dist < 6
    if obj.type == "CAMERA":
        entry["focal_length"] = round(obj.data.lens, 1)
    if obj.type == "ARMATURE":
        entry["note"] = "hero character — check that lights with dist_to_hero < 6 have enough energy (>300) to illuminate it"
    snapshot.append(entry)

snapshot.append({"_hero_location": hero_loc, "_note": "lights with likely_hits_hero=true and energy<200 will leave the character dark"})
result = json.dumps(snapshot, indent=2)
"""


def run(
    original_prompt: str,
    scene_spec: dict,
    scene_snapshot: str,
    client: anthropic.Anthropic,
    iteration: int = 1,
) -> dict:
    """
    Score the current scene. Returns dict: score, verdict, strengths, issues,
    feedback_for_orchestrator, immediate_fixes.
    """
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Original prompt: {original_prompt}\n\n"
                    f"Scene spec used:\n{json.dumps(scene_spec, indent=2)}\n\n"
                    f"Current Blender scene objects:\n{scene_snapshot}\n\n"
                    f"QA iteration: {iteration}. Score and review this scene."
                ),
            }
        ],
    )

    raw = message.content[0].text.strip()

    review_match = re.search(r"<review>(.*?)</review>", raw, re.DOTALL)
    fixes_match = re.search(r"<fixes>(.*?)</fixes>", raw, re.DOTALL)

    if not review_match:
        raise ValueError(f"Missing <review> tag in QA response:\n{raw[:500]}")

    result = json.loads(review_match.group(1).strip())

    fixes_raw = fixes_match.group(1).strip() if fixes_match else ""
    result["immediate_fixes"] = [
        f.strip() for f in fixes_raw.split("===") if f.strip()
    ] if fixes_raw else []

    return result


def get_snapshot_code() -> str:
    return SCENE_SNAPSHOT_CODE.strip()
