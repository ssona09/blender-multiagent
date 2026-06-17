"""
Composition Agent
Receives the structured scene spec from the Orchestrator and outputs:
  - bpy Python commands to build geometry and position the camera
  - a JSON summary of what was placed (fed to Lighting Agent)
"""

import anthropic
import json
import re


SYSTEM_PROMPT = """You are a Blender scene layout expert. You receive a structured JSON scene spec.

Respond in this exact format — two tagged sections, nothing outside them:

<summary>
{"objects_placed":["name1","name2"],"camera_location":[0,-6,2],"camera_rotation_deg":[80,0,0],"scene_bounds":"one line only no newlines"}
</summary>
<commands>
# --- COMMAND 1: clear scene ---
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
result = "done"
===
# more command blocks separated by ===
</commands>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OBJECT COMPOSITION — build real shapes from multiple primitives
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each command block can add several primitives to form ONE recognisable object.
Think like a prop artist — combine simple shapes into convincing silhouettes.

STREET LAMP example:
# --- COMMAND N: street lamp ---
import math
# pole
bpy.ops.mesh.primitive_cylinder_add(radius=0.04, depth=4.0, location=(3, 0, 2.0))
pole = bpy.context.object
pole.name = "Lamp_Pole"
# horizontal arm
bpy.ops.mesh.primitive_cylinder_add(radius=0.03, depth=0.8, location=(3.4, 0, 4.1))
arm = bpy.context.object
arm.name = "Lamp_Arm"
arm.rotation_euler = (0, math.radians(90), 0)
# lamp head — wide base faces DOWN (default cone orientation, no rotation needed)
bpy.ops.mesh.primitive_cone_add(radius1=0.18, radius2=0.05, depth=0.22, location=(3.8, 0, 3.98))
shade = bpy.context.object
shade.name = "Lamp_Shade"
# rotation_euler stays (0,0,0): radius1 (wide) naturally faces -Z = downward
# bulb
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.07, location=(3.8, 0, 3.85))
bulb = bpy.context.object
bulb.name = "Lamp_Bulb"
result = "done"

BUILDING example (layered silhouette — NO window objects):
# --- COMMAND N: buildings background ---
# Tall main tower
bpy.ops.mesh.primitive_cube_add(size=1, location=(-5, 8, 6))
b = bpy.context.object; b.name = "Tower_A"; b.scale = (2.0, 1.5, 6.0)
# Shorter block beside it
bpy.ops.mesh.primitive_cube_add(size=1, location=(-2.5, 9, 3.5))
b = bpy.context.object; b.name = "Tower_B"; b.scale = (1.2, 1.2, 3.5)
# Setback rooftop block (gives NYC layered look)
bpy.ops.mesh.primitive_cube_add(size=1, location=(-5, 8, 11.5))
b = bpy.context.object; b.name = "Tower_A_Top"; b.scale = (1.0, 0.8, 1.5)
result = "done"

⚠️  DO NOT add individual window cubes — floating rectangles look broken without materials.
    Give buildings visual interest through VARIED HEIGHTS and SETBACKS instead.

BENCH example:
# --- COMMAND N: park bench ---
import math
# seat
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 2, 0.45))
seat = bpy.context.object
seat.name = "Bench_Seat"
seat.scale = (0.9, 0.22, 0.04)
# back rest
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 2.18, 0.68))
back = bpy.context.object
back.name = "Bench_Back"
back.scale = (0.9, 0.03, 0.2)
# four legs (cylinders)
for x in [-0.7, 0.7]:
    for y in [1.85, 2.15]:
        bpy.ops.mesh.primitive_cylinder_add(radius=0.025, depth=0.45, location=(x, y, 0.22))
        leg = bpy.context.object
        leg.name = f"Bench_Leg_{x}_{y}"
result = "done"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ALLOWED PRIMITIVES (use these, nothing else):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- bpy.ops.mesh.primitive_plane_add(size=N, location=(x,y,z))
- bpy.ops.mesh.primitive_cube_add(size=N, location=(x,y,z))
- bpy.ops.mesh.primitive_cylinder_add(radius=N, depth=N, location=(x,y,z))
- bpy.ops.mesh.primitive_uv_sphere_add(radius=N, location=(x,y,z))
- bpy.ops.mesh.primitive_cone_add(radius1=N, radius2=N, depth=N, location=(x,y,z))
- bpy.ops.mesh.primitive_torus_add(major_radius=N, minor_radius=N, location=(x,y,z))
- bpy.ops.object.empty_add(location=(x,y,z))
- bpy.ops.object.camera_add(location=(x,y,z))
- obj.location, obj.scale, obj.rotation_euler, obj.location.x/y/z
- import math  (for math.radians, math.pi)
- Python for loops to repeat elements (windows, fence posts, etc.)

FORBIDDEN — these crash Blender:
- bpy.ops.object.particle_system_add / any particle ops
- bpy.ops.object.modifier_add
- bpy.data.materials or any node trees
- bpy.ops.curve.* or bpy.ops.mesh.* other than primitive_*_add
- bpy.context.view_layer

STREET SCENE LAYOUT — use this coordinate system for any street/alley scene:
  • Camera sits at roughly (0, -6, 2) looking toward +Y
  • Street runs along the Y axis (depth goes from Y=0 near to Y=20 far)
  • LEFT building wall:  X = -5 to -8,  depth Y = 0 to 20,  tall Z scale 4–8
  • RIGHT building wall: X = +5 to +8,  depth Y = 0 to 20,  tall Z scale 4–8
  • Street props (lamps, vents, hero character) sit between X=-3 and X=3
  • Buildings MUST be close enough to frame the shot — if X is too large they'll be off-screen
  • Use multiple cube blocks per side at different Y offsets for a layered cityblock look

RULES:
1. After any primitive_*_add(), the new object is immediately bpy.context.object. Name it right away.
2. Always end with: bpy.context.scene.camera = cam  (in the camera block)
3. Each block ends with: result = "done"
4. Separate blocks with === on its own line.
5. Max 20 command blocks. Group one full object (all its sub-parts) per block.
6. No markdown fences. No text outside the two tags.
"""


def _parse(raw: str) -> dict:
    summary_match = re.search(r"<summary>(.*?)</summary>", raw, re.DOTALL)
    commands_match = re.search(r"<commands>(.*?)</commands>", raw, re.DOTALL)

    if not summary_match or not commands_match:
        print(f"\n[DEBUG] Parse failed. Response length={len(raw)} chars. Full response:\n{raw}\n")
        raise ValueError(f"Missing <summary> or <commands> tags (response={len(raw)} chars)")

    # Parse summary JSON — replace literal newlines inside strings with space
    summary_raw = summary_match.group(1).strip()
    summary_raw = re.sub(r':\s*"([^"]*)\n([^"]*)"', lambda m: ': "' + m.group(1) + ' ' + m.group(2) + '"', summary_raw)
    try:
        summary = json.loads(summary_raw)
    except json.JSONDecodeError:
        # Fallback: build a minimal summary so we don't abort
        summary = {"objects_placed": [], "camera_location": [0, -6, 2], "camera_rotation_deg": [80, 0, 0], "scene_bounds": "unknown"}

    commands_raw = commands_match.group(1).strip()
    commands = [c.strip() for c in commands_raw.split("===") if c.strip()]

    return {"summary": summary, "bpy_commands": commands}


CHARACTER_IMPORT_CODE = """
# --- COMMAND: import hero character ---
import bpy, os, traceback

fbx_path = "{fbx_path}"

if not os.path.exists(fbx_path):
    result = f"ERROR: file not found at {{fbx_path}}"
else:
    try:
        before = set(bpy.data.objects.keys())
        bpy.ops.wm.fbx_import(filepath=fbx_path)
        # Use bpy.data (always current) not bpy.context.selected_objects (stale in timer)
        imported = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before]

        roots = [o for o in imported if o.type == 'ARMATURE' and o.parent is None]
        if not roots:
            roots = [o for o in imported if o.parent is None]

        info = []
        for root in roots:
            # Mixamo always exports at cm scale (~170 units tall) — always scale 0.01
            root.scale = (0.01, 0.01, 0.01)
            root.location = ({x}, {y}, {z})
            root.rotation_euler.z = {rz}
            info.append(root.name)

        result = f"OK: {{len(imported)}} objects imported, scaled roots: {{info}}"
    except Exception:
        result = f"ERROR:\\n{{traceback.format_exc()}}"
"""

CHARACTER_FBX = "/Users/sonali/Downloads/character.fbx"
CHARACTER_DAE = "/Users/sonali/Downloads/Talking on Phone/Talking On Phone.dae"


def run(scene_spec: dict, client: anthropic.Anthropic, include_character: bool = True) -> dict:
    """
    Generate bpy commands to build scene geometry + camera.
    Pass include_character=False to skip the FBX import (e.g. for demos or non-street scenes).
    """
    char_note = (
        "A real Mixamo character will be imported at (0,0,0) — leave that spot clear, do NOT build a humanoid."
        if include_character
        else "No character will be imported — feel free to use the center of the scene for props."
    )

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": (
                    f"Scene spec:\n{json.dumps(scene_spec, indent=2)}\n\n"
                    "Build this scene with DETAILED composed objects — not simple single cubes. "
                    "Every hero prop (lamp post, sign, vehicle, etc.) should be assembled from multiple primitives. "
                    "BUILDINGS: use varied-height layered cube blocks for silhouette depth. "
                    "DO NOT add individual window objects — they look broken without materials. "
                    "Use for-loops only for structural repeats like lamp posts or fence rails, NOT windows. "
                    "Limit to 8 distinct objects max — quality over quantity. "
                    "Keep the <summary> objects_placed list SHORT (group related items, e.g. 'street_lamps x2'). "
                    f"{char_note} "
                    "Generate the <summary> then ALL <commands> before stopping."
                ),
            }
        ],
    )

    raw = message.content[0].text.strip()
    result = _parse(raw)

    if include_character:
        import_cmd = CHARACTER_IMPORT_CODE.format(
            fbx_path=CHARACTER_FBX,
            x=0, y=0, z=0,
            rz=0,
        ).strip()
        cmds = result["bpy_commands"]
        result["bpy_commands"] = cmds[:-1] + [import_cmd] + [cmds[-1]]
        result["summary"]["objects_placed"].append("HeroCharacter (FBX)")

    return result
