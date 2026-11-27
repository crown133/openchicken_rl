# OpenChicken appearance

The MJCF variants and the source head-only URDF share this palette:

| Material | sRGB hex | Parts |
| --- | --- | --- |
| Warm white | #F1EFE8 | Upper shell, body covers, thigh covers, soles |
| Vermillion | #D9362B | Comb, feet, ankle covers |
| Terracotta | #D96A32 | Bottom head shell |
| Beak gold | #EFC477 | Upper and lower beak |
| Charcoal | #292D2E | Eye frame, neck, servos, leg structure |
| Silver | #A6ADB0 | Bearings and exposed metal fittings |

MJCF uses low specular/shininess for the plastic parts. URDF defines base RGBA; shading depends on the viewer.

`assets/visual/` contains display-only components separated from the original upper-head STL along existing disconnected component boundaries. The shell, upper beak, eye frame and internal components retain every original triangle. Original meshes remain the collision sources. The matching URDF visual meshes are under `source/urdf/meshes/visual/`; its comb is separated from the original combined upper-head visual. No lens geometry or wattles have been added.

The original URDF describes the head assembly only; its joints, inertias and collision geometry are unchanged. Body and foot colors apply to the complete MJCF models. The source URDF contains a pre-existing duplicate `xl330_neck_yaw_fixed` joint, which prevents direct loading; this appearance-only update preserves that joint structure.

The comb is visual-only in every MJCF collision variant. The URDF upper-head collision uses `meshes/collision/openchicken_upper_head.stl` (the runtime upper head without the comb); its original combined head STL is retained as source data. Comb appearance and all body mass/inertia values are preserved.

The `neck` and `neck_pitch` bodies are visual-only for collision purposes in every MJCF variant: their neck supports, two XL330 housings, and the 22x16x4 neck bearing have no collision geoms. Visual meshes, explicit inertias, joints, and actuators are retained. The separate `yaw_roll_motion` body keeps its existing collisions. The head-only source URDF has no `neck` or `neck_pitch` links, so it has no corresponding collisions to remove. The retained Onshape export configs already omit these neck collisions; they are upstream export inputs, not a full regeneration recipe for the migrated head assembly.

The four `allcollisions` variants use two convex meshes per `hip_l` bracket:
`assets/collision/hip_l_stem.stl` and `assets/collision/hip_l_end.stl`.
Both come from the original mesh split at x = 0.024225025903433565 m,
including triangle-edge/plane intersections before taking each convex hull.
There are no primitive shapes mixed into either hip bracket collision.

All standalone bearing meshes (`seeed_bearing__configuration__22x16x4` and
`seeed_bearing__configuration_default`) are visual-only in every model.
The remaining ten bearing collision geoms were removed from each of the four
allcollisions variants; the other variants and head-only URDF had none.
`bearing_roll` is the bearing support bracket, not the ring bearing itself;
its existing collision remains. Bearing visuals, rigid-body masses/inertias,
joint limits, self-collision masks and FOLD keyframes are unchanged.
Collision counts are 42 for allcollisions and 48 for allcollisions_rollers,
with the same counts in their backlash twins.
SIT scene keyframes use head_pitch = 1.55 rad (both qpos and ctrl), within the
existing +/- pi/2 range. This does not change sitstand training targets.
