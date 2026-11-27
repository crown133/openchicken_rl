"""Embed a checkpoint's reference in an already-normalized Supertracking ONNX.

Only the latent lookup is added; actor weights and normalization stay untouched.
The caller supplies chronological sensor history and an int64 frame counter.
"""
from pathlib import Path
import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


SCHEMA = "openchicken_supertracking_embedded_v1"


def sha256(path: Path) -> str:
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def embed_motion_reference(policy: Path, latent_file: Path) -> onnx.ModelProto:
    """Return history[B,10,48], time_step[B,1] -> action[B,14].

    time_step is int64, measured in 50 Hz frames. Negative values map to frame
    zero; nonnegative values wrap modulo the reference length, as in training.
    This graph is stateless: repeated calls do not advance the counter.
    """
    policy, latent_file = Path(policy), Path(latent_file)
    with np.load(latent_file, allow_pickle=False) as archive:
        latent = archive["latent"].copy()
        fps = archive["fps"]
    if (latent.ndim != 2 or latent.shape[1] != 32 or len(latent) == 0
            or latent.dtype != np.float32 or not np.isfinite(latent).all()):
        raise ValueError("Expected a finite float32 latent table of shape [frames,32]")
    if np.asarray(fps).size != 1 or float(np.asarray(fps).item()) != 50:
        raise ValueError("Supertracking reference must run at 50 Hz")

    model = onnx.load(str(policy))
    onnx.external_data_helper.convert_model_from_external_data(model)
    tensors = {value.name: value for value in model.graph.input}
    if set(tensors) != {"history", "motion_latent"}:
        raise ValueError("Expected the original history + motion_latent Supertracking interface")
    for name, trailing in (("history", [10, 48]), ("motion_latent", [32])):
        tensor = tensors[name].type.tensor_type
        if (tensor.elem_type != TensorProto.FLOAT
                or len(tensor.shape.dim) != len(trailing) + 1
                or [d.dim_value for d in tensor.shape.dim[1:]] != trailing):
            raise ValueError(f"Invalid {name} input shape or type")
    if len(model.graph.output) != 1 or model.graph.output[0].name != "action":
        raise ValueError("Expected a single action output")
    output = model.graph.output[0].type.tensor_type
    if output.elem_type != TensorProto.FLOAT or len(output.shape.dim) != 2 or output.shape.dim[1].dim_value != 14:
        raise ValueError("Expected float32 action[batch,14]")
    opsets = {item.domain: item.version for item in model.opset_import}
    if opsets.get("", 0) < 13:
        raise ValueError("Embedding requires ONNX opset 13 or newer")

    prefix = "embedded_reference/"
    names = {v.name for v in (*model.graph.input, *model.graph.output, *model.graph.initializer)}
    for node in model.graph.node:
        names.update(node.input)
        names.update(node.output)
    if "time_step" in names or any(name.startswith(prefix) for name in names):
        raise ValueError("Embedded reference names already exist in this graph")
    constants = {
        "latent": latent,
        "zero": np.array(0, dtype=np.int64),
        "length": np.array(len(latent), dtype=np.int64),
        "axis": np.array([1], dtype=np.int64),
    }
    model.graph.initializer.extend(numpy_helper.from_array(value, prefix + key)
                                   for key, value in constants.items())
    nodes = [
        helper.make_node("Max", ["time_step", prefix + "zero"], [prefix + "nonnegative"]),
        helper.make_node("Mod", [prefix + "nonnegative", prefix + "length"], [prefix + "wrapped"], fmod=0),
        helper.make_node("Squeeze", [prefix + "wrapped", prefix + "axis"], [prefix + "index"]),
        helper.make_node("Gather", [prefix + "latent", prefix + "index"], ["motion_latent"], axis=0),
    ]
    nodes.extend(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes)
    history = tensors["history"]
    batch_dim = history.type.tensor_type.shape.dim[0]
    batch = batch_dim.dim_param or (batch_dim.dim_value if batch_dim.HasField("dim_value") else None)
    del model.graph.input[:]
    model.graph.input.extend([history, helper.make_tensor_value_info("time_step", TensorProto.INT64, [batch, 1])])
    metadata = {item.key: item.value for item in model.metadata_props}
    metadata.pop("motion_latents_file", None)
    metadata.update(
        policy_schema=SCHEMA,
        command_names="time_step",
        input_shapes=json.dumps({"history": [batch, 10, 48], "time_step": [batch, 1]}),
        input_dtypes=json.dumps({"history": "float32", "time_step": "int64"}),
        reference_storage="embedded ONNX initializer; no external motion file required",
        reference_frames=str(len(latent)),
        control_frequency_hz="50",
        playback_mode="loop",
        time_step_semantics="frame_index=max(time_step,0)%reference_frames; caller advances at 50 Hz and resets to 0",
        history_order="oldest to newest, including current observation",
        normalization="History RunningNorm embedded; encoded motion already includes goal normalization",
        source_onnx_sha256=sha256(policy),
        embedded_latent_file_sha256=sha256(latent_file),
        runtime_61D_compatible="false",
    )
    helper.set_model_props(model, metadata)
    onnx.checker.check_model(model, full_check=True)
    return model


def validate_embedded_reference(source: Path, embedded: onnx.ModelProto, latent_file: Path) -> dict:
    """Compare every phase and boundary indices against the original ONNX."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    original = ort.InferenceSession(str(source), sess_options=options, providers=["CPUExecutionProvider"])
    converted = ort.InferenceSession(embedded.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"])
    with np.load(latent_file, allow_pickle=False) as archive:
        latent = archive["latent"]
    length = len(latent)
    rng = np.random.default_rng(20261103)
    # Include clip boundaries, restart, negative input, large counters and batch=1.
    groups = [np.array([0], dtype=np.int64),
              np.array([length - 1, length, length + 1, 0, -1, 2 * length,
                        np.iinfo(np.int64).max], dtype=np.int64)]
    groups += list(np.array_split(np.arange(length, dtype=np.int64), max(1, (length + 127) // 128)))
    source_batch = original.get_inputs()[0].shape[0]
    if isinstance(source_batch, int):
        raise ValueError("Validation currently requires a dynamic batch Supertracking export")
    maximum_error = 0.
    comparisons = 0
    for scale in (0., 1., 100.):
        for steps in groups:
            history = (rng.standard_normal((len(steps), 10, 48)) * scale).astype(np.float32)
            reference = latent[np.maximum(steps, 0) % length]
            expected = original.run(["action"], {"history": history, "motion_latent": reference})[0]
            inputs = {"history": history, "time_step": steps[:, None]}
            actual = converted.run(["action"], inputs)[0]
            np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
            assert np.isfinite(actual).all()
            # The graph must never auto-increment a hidden clock between calls.
            np.testing.assert_array_equal(actual, converted.run(["action"], inputs)[0])
            maximum_error = max(maximum_error, float(np.max(np.abs(actual - expected))))
            comparisons += len(steps)
    return {"onnx_checker": "passed", "reference_frames": length,
            "compared_actions": comparisons, "max_abs_error": maximum_error,
            "all_phases_checked": True, "loop_reset_negative_large_counter_checked": True,
            "repeated_call_is_stateless": True}
