"""Coordinate and seam checks for the offline motion preparation tools."""
import importlib.util
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest


def test_bundled_reference_paths_and_hashes():
    from mjlab_microduck.tasks.microduck_supertracking_env_cfg import DEFAULT_MOTION, ROBUST_MOTION
    root = Path(__file__).resolve().parents[1]
    for name in (DEFAULT_MOTION, ROBUST_MOTION):
        path = Path(name)
        assert path.is_relative_to(root / 'motion/assets')
        manifest = json.loads(path.with_name('manifest.json').read_text())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['sha256']


@pytest.fixture(scope="module")
def preparation():
    path = Path(__file__).resolve().parents[1] / "scripts/prepare_supertracking_motion.py"
    spec = importlib.util.spec_from_file_location("supertracking_preparation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def simple_model():
    return mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><freejoint/><geom type="sphere" size=".02" mass="1"/>
        <body pos="0 0 .05"><joint name="hinge" axis="0 1 0"/>
          <geom type="sphere" size=".01" mass=".1"/></body>
      </body></worldbody></mujoco>''')


def test_periodic_velocity_has_correct_units_and_no_seam_spike(preparation, simple_model):
    time = np.arange(501) / 50
    phase = 2 * np.pi * .6 * time  # Six exact cycles; not an integer frames/cycle.
    poses = np.tile(simple_model.qpos0, (len(time), 1))
    poses[:, 1] = .01 * np.sin(phase)
    angle = .2 * np.sin(phase)
    poses[:, 3], poses[:, 6] = np.cos(angle / 2), np.sin(angle / 2)
    poses[:, 7] = .3 * np.sin(phase)
    velocity = preparation.differentiate(simple_model, poses, .02)
    expected = 2 * np.pi * .6 * np.cos(phase)
    np.testing.assert_allclose(velocity[:, 1], .01 * expected, atol=4e-5)
    np.testing.assert_allclose(velocity[:, 5], .2 * expected, atol=8e-4)
    np.testing.assert_allclose(velocity[:, 6], .3 * expected, atol=.0012)
    np.testing.assert_array_equal(velocity[0], velocity[-1])


def test_nonclosed_clip_is_not_silently_wrapped(preparation, simple_model):
    poses = np.tile(simple_model.qpos0, (3, 1))
    poses[-1, 1] = .01
    with pytest.raises(ValueError, match="closed clip"):
        preparation.differentiate(simple_model, poses, .02)


def test_training_model_retains_battery_and_leg_collision_geometry(preparation):
    model, joints, bodies = preparation.training_model()
    assert len(joints) == 14 and bodies[0] == "trunk_base"
    found = set()
    for geom in range(model.ngeom):
        mesh = model.geom_dataid[geom]
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH and model.geom_contype[geom]:
            found.add((model.body(model.geom_bodyid[geom]).name, model.mesh(mesh).name))
    assert ("trunk_base", "np_f970") in found
    assert ("leg", "leg") in found
    assert model.geom("audit_floor").contype[0] == 1


def test_v12_downbeat_stance_repair_removes_battery_collision(preparation, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    from repair_supertracking_collision import repair_cycle

    # Original accepted V12 downbeat, t=1.12 s: left shin penetrates the battery.
    pose = np.array([
        .00280256369340574, -.003293846719325564, .08266492496376075,
        .974776139059341, .16125803551621157, -.016840468163608995, -.15337445463325064,
        -.3307272528010995, -.3665191429188082, .3207283464119991,
        .9243791766369199, .6919204296297683, .255090279566987,
        .3412877165663836, .2979699264727465, -.06981317007977318,
        -.3211736184386107, -.26217701298892526, -.2960704695938232,
        -1.2775379056472445, -1.0667788229218869,
    ])
    model, names, _ = preparation.training_model()
    data = mujoco.MjData(model)
    data.qpos[:] = pose
    mujoco.mj_forward(model, data)
    assert min(c.dist for c in data.contact) < -.007
    foot = model.body("ankle_left").id
    before_position = data.xpos[foot].copy()
    before_rotation = data.xmat[foot].copy()
    fixed = repair_cycle(model, names, pose[None])[0]
    left = [model.joint(n).qposadr[0] for n in names if n.startswith("left_")]
    unchanged = np.setdiff1d(np.arange(model.nq), left)
    np.testing.assert_array_equal(fixed[unchanged], pose[unchanged])
    data.qpos[:] = fixed
    mujoco.mj_forward(model, data)
    assert all(c.dist > -.0001 for c in data.contact)
    np.testing.assert_allclose(data.xpos[foot] - before_position, [.025, 0, 0], atol=3e-7)
    np.testing.assert_allclose(data.xmat[foot], before_rotation, atol=1e-6)
