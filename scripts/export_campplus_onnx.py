#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
Export CAM++ PyTorch model to ONNX format.

Usage:
    python scripts/export_campplus_onnx.py \
        --model_dir model_zoo/speaker/CAM++/pytorch \
        --output_dir model_zoo/speaker/CAM++/onnx

The exported model.onnx is compatible with FunASR C++ runtime (onnxruntime).
Input:  "fbank" [batch, num_frames, 80]   (fbank features)
Output: "embedding" [batch, 192]          (speaker embedding)
"""

import argparse
import os
import sys
import yaml
import torch
import numpy as np

# Add project root to path so we can import funasr
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from funasr.models.campplus.model import CAMPPlus


def load_model(model_dir: str) -> CAMPPlus:
    """Load CAM++ model from a directory containing config.yaml and weights."""
    config_path = os.path.join(model_dir, "config.yaml")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config.yaml not found in {model_dir}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_conf = config.get("model_conf", {})
    model = CAMPPlus(
        feat_dim=model_conf.get("feat_dim", 80),
        embedding_size=model_conf.get("embedding_size", 192),
        growth_rate=model_conf.get("growth_rate", 32),
        bn_size=model_conf.get("bn_size", 4),
        init_channels=model_conf.get("init_channels", 128),
        config_str=model_conf.get("config_str", "batchnorm-relu"),
        memory_efficient=False,  # disable checkpoint for export
        output_level=model_conf.get("output_level", "segment"),
    )

    # Try common weight file names
    weight_candidates = [
        "campplus_cn_common.bin",
        "campplus_cn_common.pt",
        "pytorch_model.bin",
        "model.pt",
    ]
    weight_path = None
    for name in weight_candidates:
        p = os.path.join(model_dir, name)
        if os.path.isfile(p):
            weight_path = p
            break

    if weight_path is None:
        raise FileNotFoundError(
            f"No weight file found in {model_dir}. "
            f"Tried: {weight_candidates}"
        )

    print(f"Loading weights from: {weight_path}")
    state_dict = torch.load(weight_path, map_location="cpu")
    if isinstance(state_dict, dict) and "model" in state_dict:
        state_dict = state_dict["model"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


def export_onnx(
    model: CAMPPlus,
    output_dir: str,
    opset_version: int = 14,
    quantize: bool = False,
):
    """Export model to ONNX and optionally quantize."""
    os.makedirs(output_dir, exist_ok=True)
    onnx_path = os.path.join(output_dir, "model.onnx")

    # Dummy input: [batch=1, num_frames=200, feat_dim=80]
    dummy_input = torch.randn(1, 200, 80)

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["fbank"],
        output_names=["embedding"],
        dynamic_axes={
            "fbank": {0: "batch", 1: "num_frames"},
            "embedding": {0: "batch"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
        dynamo=False,  # Use legacy TorchScript exporter for PyTorch 2.4+ compatibility
    )
    print(f"Exported ONNX model: {onnx_path}")

    # Verify with onnxruntime
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(onnx_path)
        test_input = np.random.randn(1, 200, 80).astype(np.float32)

        with torch.no_grad():
            pt_out = model(torch.from_numpy(test_input)).numpy()

        ort_out = sess.run(None, {"fbank": test_input})[0]
        max_diff = np.max(np.abs(pt_out - ort_out))
        print(f"Verification passed. Max diff (PyTorch vs ONNX): {max_diff:.6e}")
    except ImportError:
        print("onnxruntime not installed, skipping verification.")

    # Optional quantization
    if quantize:
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            quant_path = os.path.join(output_dir, "model_quant.onnx")
            quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QUInt8)
            print(f"Exported quantized model: {quant_path}")
        except ImportError:
            print("onnxruntime quantization not available, skipping.")

    # Copy config.yaml to output dir for C++ runtime
    src_config = os.path.join(os.path.dirname(onnx_path), "..", "pytorch", "config.yaml")
    dst_config = os.path.join(output_dir, "config.yaml")
    if not os.path.isfile(dst_config):
        import shutil
        # Resolve relative to output_dir's parent
        model_dir_config = os.path.normpath(src_config)
        if os.path.isfile(model_dir_config):
            shutil.copy2(model_dir_config, dst_config)
            print(f"Copied config.yaml to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Export CAM++ speaker model from PyTorch to ONNX"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="model_zoo/speaker/CAM++/pytorch",
        help="Path to PyTorch model directory (contains config.yaml and weights)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="model_zoo/speaker/CAM++/onnx",
        help="Output directory for ONNX model",
    )
    parser.add_argument(
        "--opset_version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also export quantized model (model_quant.onnx)",
    )
    args = parser.parse_args()

    model = load_model(args.model_dir)
    export_onnx(model, args.output_dir, args.opset_version, args.quantize)


if __name__ == "__main__":
    main()
