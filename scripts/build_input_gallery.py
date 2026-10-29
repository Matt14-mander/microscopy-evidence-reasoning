"""Build a local, full-resolution-linked gallery from an existing input audit (no model)."""

import argparse
import html
from pathlib import Path

from src.utils import read_json


def build_gallery(directory):
    directory = Path(directory)
    report = read_json(directory / "audit.json")
    if report["status"] != "passed":
        raise ValueError("Only a passed input audit can be published as a gallery")
    sections = []
    for sample_id in report["cases"]:
        items = [im for im in report["images"] if im["sample_id"] == sample_id]
        blocks = []
        for item in items:
            preview = item["preview"]
            stem = Path(preview).stem.removesuffix("-preview")
            links = []
            for suffix, label in (("source-thumbnail", "原图缩略图"), ("study", "公共研究输入"),
                                  ("baseline", "实际 baseline"), ("blur", "实际 blur"), ("resolution", "实际 resolution")):
                path = f"images/{stem}-{suffix}.png"
                if not (directory / path).is_file():
                    raise FileNotFoundError(path)
                links.append(f'<a href="{path}">{label}</a>')
            metrics = "；".join(f'{name}: RGB MAE={v["rgb_mae_0_1"]:.6f}, 梯度比={v["gradient_ratio"]:.3f}'
                                if v["gradient_ratio"] is not None else f'{name}: 均匀基线，梯度比不可定义'
                                for name, v in item["metrics"].items())
            blocks.append(f'<h3>第 {item["image_index"] + 1} 张图</h3><p>{" · ".join(links)}</p>'
                          f'<a href="{preview}"><img loading="lazy" src="{preview}" alt="五阶段输入对照"></a>'
                          f'<p class="metrics">{metrics}</p>')
        sections.append(f'<details><summary>题目 {html.escape(sample_id)} · {len(items)} 张图</summary>{"".join(blocks)}</details>')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>MicroVQA pilot50 实际输入审计</title><style>body{font:16px system-ui;max-width:1800px;margin:24px auto;padding:16px;line-height:1.6}'
            'summary{padding:14px;cursor:pointer;background:#eef5f7}details{border:1px solid #ddd;margin:12px 0;padding:8px}img{width:100%;height:auto}'
            'a{color:#176278}.metrics{font-family:monospace;font-size:14px}h1{font-size:28px}</style>'
            '<h1>MicroVQA pilot50：实际模型输入预览</h1>'
            '<p>开发集 50 题 / 45 个图像组 / 90 张图片。点击题目展开；每张图可分别打开实际尺寸 PNG。</p>'
            '<p>从左到右：原始图（展示缩放）、公共研究输入、实际 baseline、实际 blur、实际 resolution。'
            '后三者从 processor 的 pixel_values 反向重建，反归一化后舍入为 uint8。拼图会适应屏幕；链接 PNG 保留实际尺寸。'
            'Text-only 不传图片，也没有空白替代图。</p>'
            '<p>RGB MAE 与梯度比仅描述像素变化，不证明关键生物学证据受损；暗背景会稀释全图平均差异。'
            '本页不展示正确答案或预测结果。</p>'
            + ('<p>全部图像的退化诊断：<a href="diagnostics.png">PNG</a> · <a href="diagnostics.pdf">PDF</a></p>'
               if (directory / "diagnostics.pdf").exists() else '')
            + "".join(sections) + '</html>')
    path = directory / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", default="outputs/input-audit/pilot50-v1")
    print(build_gallery(parser.parse_args().directory))
