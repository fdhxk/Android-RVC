# -*- coding: utf-8 -*-
"""
ONNX Runtime 推理封装（手机端）

用 onnxruntime 替代 PyTorch 执行 RVC 的两个大模型前向，提供与
PyTorch 模型接口兼容的 wrapper，使 RVC 官方 Pipeline 代码无需改动。

依赖: onnxruntime（Linux aarch64 下自动启用 XNNPACK EP 加速，
      不支持的算子自动回退到 CPU EP）

模型文件（由 PC 端 export_onnx.py 导出）:
  assets/hubert/hubert_layer9.onnx    (RVC v1 特征提取)
  assets/hubert/hubert_layer12.onnx   (RVC v2 特征提取)
  models/<模型名>.onnx                (Synthesizer)
"""
import logging
import math
import os

import numpy as np
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)

# XNNPACK 在前，不支持算子自动落到 CPU EP
_PROVIDERS = ["XnnpackExecutionProvider", "CPUExecutionProvider"]


def _make_session_options(threads):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = max(1, threads)
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts


def _create_session(path, opts):
    """创建 ORT session，XNNPACK 不可用时自动回退纯 CPU"""
    try:
        return ort.InferenceSession(path, sess_options=opts, providers=_PROVIDERS)
    except Exception:
        logger.warning("XNNPACK EP 不可用，回退到 CPU EP: %s", path)
        return ort.InferenceSession(
            path, sess_options=opts, providers=["CPUExecutionProvider"]
        )


class OnnxHubertWrapper:
    """兼容 hubert_model 接口：extract_features() / final_proj()"""

    def __init__(self, model_dir, threads=4):
        self.model_dir = model_dir
        self.sessions = {}
        self.opts = _make_session_options(threads)

    def _load(self, layer):
        if layer not in self.sessions:
            path = os.path.join(self.model_dir, "hubert_layer%d.onnx" % layer)
            if not os.path.exists(path):
                raise FileNotFoundError(
                    "缺少 ONNX 模型: %s（请先用 PC 端 export_onnx.py 导出）" % path
                )
            logger.info("加载 HuBERT ONNX (layer %d): %s", layer, path)
            self.sessions[layer] = _create_session(path, self.opts)
        return self.sessions[layer]

    def extract_features(self, source, padding_mask=None, mask=False,
                         ret_conv=False, output_layer=9):
        """与 hubert_model.extract_features 兼容。

        ONNX 导出时 padding_mask 传 None，模型内部自动创建帧维度 mask，
        因此这里忽略传入的 padding_mask（波形维度与帧维度不一致，
        直接喂给 ONNX 会导致 Where 广播失败）。

        Args:
            source: [1, N] float32 (CPU tensor)
            output_layer: 9 (v1) 或 12 (v2)，决定使用哪个 ONNX 模型

        Returns:
            (feats, padding_mask): feats 为 [1, T, C] float32 torch tensor
        """
        layer = int(output_layer) if output_layer is not None else 9
        sess = self._load(layer)

        src = source.detach().cpu().numpy().astype(np.float32)
        # HuBERT conv 下采样后帧数 T 的奇偶由输入长度决定：640 倍数的输入
        # 恒得奇数帧（T=2k-1），与导出时 trace 的 pad/undo-pad 分支一致，
        # 保证 Where 等节点广播正确且输出帧数与 PyTorch 版相同。
        n = src.shape[1]
        n_pad = 640 * max(1, math.ceil(n / 640))
        if n_pad != n:
            src = np.pad(src, ((0, 0), (0, n_pad - n)), mode="constant")

        feats = sess.run(["feats"], {"source": src})[0]
        logger.info(
            "HuBERT feats shape: %s (输入 %d samples, pad 到 %d)",
            feats.shape, n, n_pad,
        )
        return torch.from_numpy(feats), None

    def final_proj(self, x):
        # final_proj 已并入 hubert_layer9.onnx，恒等返回
        return x


class OnnxSynthesizerWrapper:
    """兼容 net_g 接口：infer()，按模型是否含 f0 输入自动适配调用方式"""

    def __init__(self, onnx_path, threads=4):
        self.opts = _make_session_options(threads)
        logger.info("加载 Synthesizer ONNX: %s", onnx_path)
        self.session = _create_session(onnx_path, self.opts)
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.has_f0 = "pitch" in self.input_names

    def infer(self, *args):
        """与 net_g.infer 兼容。

        有 f0 模型: args = (phone, phone_lengths, pitch, nsff0, sid)
        无 f0 模型: args = (phone, phone_lengths, sid)

        Returns:
            (audio, x_mask, None): audio 为 [1, 1, samples] torch tensor
        """
        if self.has_f0:
            phone, phone_lengths, pitch, nsff0, sid = args[:5]
        else:
            phone, phone_lengths, sid = args[:3]
            pitch = nsff0 = None

        ort_inputs = {
            "phone": phone.detach().cpu().numpy().astype(np.float32),
            "phone_lengths": phone_lengths.detach().cpu().numpy().astype(np.int64),
            "sid": sid.detach().cpu().numpy().astype(np.int64),
        }
        if pitch is not None:
            ort_inputs["pitch"] = pitch.detach().cpu().numpy().astype(np.int64)
        if nsff0 is not None:
            ort_inputs["nsff0"] = nsff0.detach().cpu().numpy().astype(np.float32)

        audio = self.session.run(["audio"], ort_inputs)[0]  # [1, 1, samples]
        logger.info(
            "Synth infer: phone=%s lengths=%s audio=%s (%.2fs @%d)",
            list(ort_inputs["phone"].shape),
            ort_inputs["phone_lengths"].tolist(),
            audio.shape,
            audio.shape[2] / 40000,
            self.session.get_inputs()[0].type,
        )
        return torch.from_numpy(audio), None, None
