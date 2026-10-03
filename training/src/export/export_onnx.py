"""
ONNX export and verification for RPi5 deployment.
"""
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 64, 100),
    opset_version: int = 17,
    dynamic_axes: Optional[dict] = None,
    simplify: bool = True,
    verify: bool = True,
    device: str = "cpu",
) -> str:
    """
    Export PyTorch model to ONNX.
    
    Args:
        model: Trained model (will be put in eval mode)
        output_path: Where to save the .onnx file
        input_shape: Example input shape (batch, channels, n_mels, time)
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axis specification
        simplify: Run onnx-simplifier
        verify: Verify ONNX output matches PyTorch
        device: Device to run export on
    
    Returns:
        Path to exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(*input_shape, device=device)
    
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 3: "time"},
            "output": {0: "batch_size"},
        }
    
    # Reference output for verification
    with torch.no_grad():
        torch_output = model(dummy_input).cpu().numpy()
    
    # Prefer the legacy TorchScript exporter (no onnxscript dependency).
    export_kwargs = dict(
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    try:
        torch.onnx.export(model, dummy_input, str(output_path), dynamo=False, **export_kwargs)
    except TypeError:
        # Older torch without the `dynamo` argument
        torch.onnx.export(model, dummy_input, str(output_path), **export_kwargs)

    print(f"Exported ONNX model to {output_path}")
    
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            onnx_model = onnx.load(str(output_path))
            simplified, check = onnx_simplify(onnx_model)
            if check:
                onnx.save(simplified, str(output_path))
                print("ONNX model simplified successfully.")
        except ImportError:
            print("onnx-simplifier not installed, skipping simplification.")
    
    if verify:
        verify_onnx(str(output_path), dummy_input.cpu().numpy(), torch_output)
    
    return str(output_path)


def verify_onnx(
    onnx_path: str,
    input_data: np.ndarray,
    reference_output: np.ndarray,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """Verify ONNX output matches PyTorch reference."""
    import onnxruntime as ort
    
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    
    onnx_output = sess.run(None, {input_name: input_data.astype(np.float32)})[0]
    
    if np.allclose(onnx_output, reference_output, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(onnx_output - reference_output))
        print(f"ONNX verification passed (max diff: {max_diff:.2e})")
        return True
    else:
        max_diff = np.max(np.abs(onnx_output - reference_output))
        print(f"WARNING: ONNX verification failed (max diff: {max_diff:.2e})")
        return False


def benchmark_onnx(
    onnx_path: str,
    input_shape: Tuple[int, ...] = (1, 1, 64, 100),
    num_runs: int = 1000,
    warmup_runs: int = 100,
    num_threads: Optional[int] = None,
) -> dict:
    """
    Benchmark ONNX inference latency.
    
    Returns dict with p50, p90, p95, p99, mean, std latencies in ms.
    """
    import time
    import onnxruntime as ort
    
    so = ort.SessionOptions()
    if num_threads is not None:
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = num_threads
    
    sess = ort.InferenceSession(
        onnx_path,
        sess_options=so,
        providers=["CPUExecutionProvider"],
    )
    input_name = sess.get_inputs()[0].name
    
    dummy = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup
    for _ in range(warmup_runs):
        sess.run(None, {input_name: dummy})
    
    # Benchmark
    latencies = []
    for _ in range(num_runs):
        start = time.perf_counter()
        sess.run(None, {input_name: dummy})
        latencies.append((time.perf_counter() - start) * 1000.0)  # ms
    
    latencies = np.array(latencies)
    
    return {
        "mean_ms": float(np.mean(latencies)),
        "std_ms": float(np.std(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p90_ms": float(np.percentile(latencies, 90)),
        "p95_ms": float(np.percentile(latencies, 95)),
        "p99_ms": float(np.percentile(latencies, 99)),
        "num_runs": num_runs,
    }


def get_onnx_model_info(onnx_path: str) -> dict:
    """Get ONNX model info (size, ops, inputs, outputs)."""
    from pathlib import Path
    import onnx
    
    path = Path(onnx_path)
    model = onnx.load(str(path))
    
    file_size_mb = path.stat().st_size / (1024 * 1024)
    
    inputs = [
        {"name": i.name, "shape": [d.dim_value if d.dim_value else d.dim_param for d in i.type.tensor_type.shape.dim]}
        for i in model.graph.input
    ]
    outputs = [
        {"name": o.name, "shape": [d.dim_value if d.dim_value else d.dim_param for d in o.type.tensor_type.shape.dim]}
        for o in model.graph.output
    ]
    
    ops = set()
    for node in model.graph.node:
        ops.add(node.op_type)
    
    return {
        "file_size_mb": file_size_mb,
        "inputs": inputs,
        "outputs": outputs,
        "op_types": sorted(ops),
        "num_nodes": len(model.graph.node),
    }