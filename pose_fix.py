import asyncio
from blender_bridge import BlenderBridge

code = """
import bpy

# Boost existing fill
fill = bpy.data.objects.get('CharFill')
if fill:
    fill.data.energy = 800

# Boost rim
rim = bpy.data.objects.get('CharRim')
if rim:
    rim.data.energy = 500

# Add a direct front key light — shines straight onto character face/chest
bpy.ops.object.light_add(type='AREA', location=(0, -2, 2.5))
key = bpy.context.object
key.name = 'CharKey'
key.rotation_euler = (1.1, 0, 0)
key.data.energy = 600
key.data.color = (1.0, 0.75, 0.9)    # soft pink-white — blends with scene neon
key.data.size = 1.5                    # large area = soft shadows

result = 'character lighting boosted'
"""


async def main():
    async with BlenderBridge() as b:
        print(await b.run(code))

asyncio.run(main())
