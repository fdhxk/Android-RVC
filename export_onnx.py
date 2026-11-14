#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RVC ONNX 导出脚本 —— 在 PC 上运行

把 RVC 的两个大模型从 PyTorch 导出为 ONNX，供手机端 onnxruntime
（XNNPACK EP）加速推理使用。f0 提取（pyworld 等 C 库）无需导出。

用法（在 android-rvc 目录下）:
    python export_onnx.py --model models/kikiV1.pth
    python export_onnx.py --model models/kikiV1.pth --out-dir /tmp/onnx

PC 端需要: python3 + torch + fairseq（与 requirements.txt 一致，CPU 即可）

输出（复制到手机对应位置）:
    <out-dir>/hubert_layer9.onnx     -> assets/hubert/
    <out-dir>/hubert_layer12.onnx    -> assets/hubert/
    <out-dir>/<模型名>.onnx          -> models/（与 .pth 同名，如 kikiV1.onnx）
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OP_SET = 17  # onnxruntime >= 1.17 均支持


def export_hubert(hubert_path, out_dir, opset=OP_SET):
    """导出 HuBERT 特征提取模型。

    - hubert_layer9: 第9层输出 + final_proj（RVC v1 用，输出 256 维）
    - hubert_layer12: 第12层输出（RVC v2 用，输出 768 维）

    输入: source [1, N] float32, padding_mask [1, N] bool（N 动态）
    输出: feats [1, T, C] float32
    """
    from infer.lib.jit.get_hubert import get_hubert_model

    model = get_hubert_model(hubert_path, torch.device("cpu"))
    model.eval()

    # N 取 640 倍数：conv 下采样后帧数 T 恒为奇数，走 pad 1 帧分支；
    # padding_mask 传 None 让模型内部自动创建帧维度 mask（形状恒正确），
    # 避免波形维度 [1, N] 的 mask 与帧维度 [1, T] 错配导致 ONNX Where 广播失败。
    # 运行时手机端 wrapper 也会把输入补齐到 640 倍数，保证分支一致。
    N = 16000 * 8
    source = torch.randn(1, N, dtype=torch.float32)
    padding_mask = torch.zeros(1, N, dtype=torch.bool)

    class HubertLayer9(torch.nn.Module):
        """RVC v1: output_layer=9 后接 final_proj"""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, source):
            logits = self.m.extract_features(
                source=source, padding_mask=None, output_layer=9
            )
            return self.m.final_proj(logits[0])

    class HubertLayer12(torch.nn.Module):
        """RVC v2: output_layer=12 直接输出"""

        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, source):
            logits = self.m.extract_features(
                source=source, padding_mask=None, output_layer=12
            )
            return logits[0]

    specs = [
        ("hubert_layer9", HubertLayer9(model)),
        ("hubert_layer12", HubertLayer12(model)),
    ]
    for name, wrapper in specs:
        path = os.path.join(out_dir, name + ".onnx")
        torch.onnx.export(
            wrapper,
            (source,),
            path,
            input_names=["source"],
            output_names=["feats"],
            dynamic_axes={
                "source": {1: "time"},
                "feats": {1: "frames"},
            },
            opset_version=opset,
        )
        print("已导出: %s" % path)


class SynthF0Wrapper(torch.nn.Module):
    """有 f0 模型的导出 wrapper（torch 2.3+ 要求导出对象是 nn.Module）"""

    def __init__(self, net_g):
        super().__init__()
        self.net_g = net_g

    def forward(self, phone, phone_lengths, pitch, nsff0, sid):
        return self.net_g.infer(phone, phone_lengths, pitch, nsff0, sid)


class SynthNoF0Wrapper(torch.nn.Module):
    """无 f0 模型的导出 wrapper"""

    def __init__(self, net_g):
        super().__init__()
        self.net_g = net_g

    def forward(self, phone, phone_lengths, sid):
        return self.net_g.infer(phone, phone_lengths, sid)


def export_synthesizer(pth_path, out_dir, opset=OP_SET):
    """导出 RVC Synthesizer（VITS 变体）推理模型。

    输入: phone [1, T, C] float32, phone_lengths [1] int64,
          pitch [1, T] int64, nsff0 [1, T] float32, sid [1] int64
          （无 f0 模型不含 pitch / nsff0）
    输出: audio [1, 1, samples] float32
    """
    from infer.lib.infer_pack.models import (
        SynthesizerTrnMs256NSFsid,
        SynthesizerTrnMs256NSFsid_nono,
        SynthesizerTrnMs768NSFsid,
        SynthesizerTrnMs768NSFsid_nono,
    )

    cpt = torch.load(pth_path, map_location=torch.device("cpu"))
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    klass = {
        ("v1", 1): SynthesizerTrnMs256NSFsid,
        ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): SynthesizerTrnMs768NSFsid,
        ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
    }[(version, if_f0)]

    net_g = klass(*cpt["config"], is_half=False)
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().float()

    dim = 768 if version == "v2" else 256
    T = 64  # 示例帧数（实际推理时动态）
    phone = torch.randn(1, T, dim, dtype=torch.float32)
    phone_lengths = torch.tensor([T], dtype=torch.long)
    sid = torch.tensor([0], dtype=torch.long)

    if if_f0 == 1:
        pitch = torch.randint(1, 256, (1, T), dtype=torch.long)
        nsff0 = torch.rand(1, T, dtype=torch.float32)
        args = (phone, phone_lengths, pitch, nsff0, sid)
        wrapper = SynthF0Wrapper(net_g)
        input_names = ["phone", "phone_lengths", "pitch", "nsff0", "sid"]
        dynamic_axes = {
            "phone": {1: "frames"},
            "pitch": {1: "frames"},
            "nsff0": {1: "frames"},
            "audio": {2: "samples"},
        }
    else:
        args = (phone, phone_lengths, sid)
        wrapper = SynthNoF0Wrapper(net_g)
        input_names = ["phone", "phone_lengths", "sid"]
        dynamic_axes = {
            "phone": {1: "frames"},
            "audio": {2: "samples"},
        }

    out_name = os.path.splitext(os.path.basename(pth_path))[0] + ".onnx"
    path = os.path.join(out_dir, out_name)
    torch.onnx.export(
        wrapper,
        args,
        path,
        input_names=input_names,
        output_names=["audio", "x_mask"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
    )
    print("已导出: %s (version=%s, f0=%d)" % (path, version, if_f0))


def main():
    parser = argparse.ArgumentParser(description="导出 RVC ONNX 模型（PC 端运行）")
    parser.add_argument(
        "--model", required=True, help="RVC .pth 模型路径，如 models/kikiV1.pth"
    )
    parser.add_argument(
        "--hubert",
        default=os.path.join("assets", "hubert", "hubert_base.pt"),
        help="hubert_base.pt 路径",
    )
    parser.add_argument("--out-dir", default="onnx_out", help="ONNX 输出目录")
    parser.add_argument("--opset", type=int, default=OP_SET)
    parser.add_argument(
        "--only",
        choices=["hubert", "synth"],
        default=None,
        help="只导出部分模型（hubert 或 synth），跳过已完成的部分",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.only in (None, "hubert"):
        print("== 导出 HuBERT ==")
        export_hubert(args.hubert, args.out_dir, args.opset)
    if args.only in (None, "synth"):
        print("\n== 导出 Synthesizer ==")
        export_synthesizer(args.model, args.out_dir, args.opset)
    print("\n完成。将 onnx_out 下的文件复制到手机对应位置：")
    print("  hubert_layer9.onnx / hubert_layer12.onnx -> assets/hubert/")
    print("  %s.onnx -> models/（与 .pth 同名）" % os.path.splitext(
        os.path.basename(args.model))[0])


if __name__ == "__main__":
    main()
