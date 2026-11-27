"""Reference embedding must preserve the trained decoder and reset/loop behavior."""
import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import pytest
import torch

from mjlab_openchicken.supertracking.export import embed_motion_reference, validate_embedded_reference
from mjlab_openchicken.supertracking.models import Actor, ExportedActor


@pytest.fixture
def bundle(tmp_path):
    torch.manual_seed(17)
    actor = Actor(torch.zeros(14), torch.tensor([[-2., 2.]] * 14)).eval()
    with torch.no_grad():
        actor.obs_norm.mean.copy_(torch.randn(48))
        actor.obs_norm.var.copy_(torch.rand(48) + .1)
        torch.nn.init.normal_(actor.decoder[-1].weight, std=.1)
    source = tmp_path / 'policy.onnx'
    torch.onnx.export(ExportedActor(actor), (torch.zeros(2, 10, 48), torch.zeros(2, 32)),
                      str(source), input_names=['history', 'motion_latent'], output_names=['action'],
                      dynamic_axes={name: {0: 'batch'} for name in ('history', 'motion_latent', 'action')},
                      opset_version=17, dynamo=False)
    model = onnx.load(source)
    onnx.helper.set_model_props(model, {'checkpoint_sha256': 'test-checkpoint',
                                      'motion_latents_file': 'motion_latents_50hz.npz'})
    onnx.save(model, source)
    latent = np.random.default_rng(19).normal(0, .2, (7, 32)).astype(np.float32)
    reference = tmp_path / 'motion_latents_50hz.npz'
    np.savez(reference, latent=latent, fps=50)
    return source, reference


def test_embedded_reference_matches_all_frames_boundaries_and_normalization(bundle, tmp_path):
    source, reference = bundle
    original = onnx.load(source)
    model = embed_motion_reference(source, reference)
    checks = validate_embedded_reference(source, model, reference)
    assert checks['max_abs_error'] < 2e-5
    assert checks['all_phases_checked'] and checks['repeated_call_is_stateless']
    initializers = {item.name: item for item in model.graph.initializer}
    for value in original.graph.initializer:
        np.testing.assert_array_equal(numpy_helper.to_array(initializers[value.name]), numpy_helper.to_array(value))
    assert [item.SerializeToString() for item in model.graph.node[4:]] == [item.SerializeToString() for item in original.graph.node]
    metadata = {item.key: item.value for item in model.metadata_props}
    assert metadata['checkpoint_sha256'] == 'test-checkpoint'
    assert 'motion_latents_file' not in metadata
    # The resulting file must still infer after both external source files vanish.
    target = tmp_path / 'embedded.onnx'
    onnx.save(model, target)
    source.unlink()
    reference.unlink()
    session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
    assert [(item.name, item.type) for item in session.get_inputs()] == [('history', 'tensor(float)'), ('time_step', 'tensor(int64)')]
    actual = session.run(None, {'history': np.zeros((1, 10, 48), np.float32),
                                'time_step': np.array([[0]], np.int64)})[0]
    assert actual.shape == (1, 14) and np.isfinite(actual).all()


@pytest.mark.parametrize('latent,fps', [
    (np.zeros((0, 32), np.float32), 50),
    (np.zeros((7, 31), np.float32), 50),
    (np.zeros((7, 32), np.float64), 50),
    (np.full((7, 32), np.nan, np.float32), 50),
    (np.zeros((7, 32), np.float32), 60),
])
def test_rejects_invalid_reference(bundle, latent, fps):
    source, reference = bundle
    np.savez(reference, latent=latent, fps=fps)
    with pytest.raises(ValueError):
        embed_motion_reference(source, reference)


def test_rejects_native_or_already_embedded_interface(bundle):
    source, reference = bundle
    model = onnx.load(source)
    model.graph.input[0].name = 'obs'
    onnx.save(model, source)
    with pytest.raises(ValueError, match='Supertracking interface'):
        embed_motion_reference(source, reference)
