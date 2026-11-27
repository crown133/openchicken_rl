# OpenChicken head URDF

This head-only description follows the rigid-body frames, mass, centre of mass,
full inertia tensor, joint axes, position limits and effort limits in
`../../robot_allcollisions.xml`. The root is `yaw_roll_motion`; upstream head_yaw,
head_pitch and neck_pitch joints belong to the full MJCF robot and are not in
this head-only export. Keep the legacy package/file name for existing references.

| Link | Mass (kg) | Contents |
| --- | ---: | --- |
| yaw_roll_motion | 0.0486 | yaw/roll bracket and yaw servo |
| bottom_head | 0.198 | head shell, internals and roll servo |
| comb | 0.001 | separate red visual mesh, no collision |
| lower_beak | 0.0026251 | independently moving beak |

Total: 0.2502251 kg. Fixed CAD parts already included in these MJCF inertias do
not receive a second mass. The old per-part CAD meshes remain as source assets.

`head_roll`: -25 to +25 degrees, effort 0.96 N m.
`passive_beak`: 0 to 30 degrees, axis -Z, effort 0.08 N m.
MJCF specifies no joint velocity limit. URDF 1.0 requires a numeric velocity;
`1000000` rad/s is an explicitly unbounded simulation placeholder (the installed
urdfdom rejects `inf`). This is not a measured hardware speed rating; select
finite controller limits separately
before hardware use. Joint damping and friction are copied from MJCF. Standard
URDF does not encode MJCF armature, position actuator gains, collision masks or
the training BAM actuator; this file is not a complete training/deployment model.

Meshes in `meshes/mjcf/` are byte-for-byte copies of the runtime MJCF assets,
kept together for a standalone head package. They resolve through
`package://20261025.SLDASM/meshes/mjcf/...` in ROS. The MuJoCo
compiler extension resolves the same local package mesh basenames and retains
visual meshes and explicit fixed-link inertia. Geometry origins use the MJCF
frames, including the small head-shell rotation and the beak hinge transform.
