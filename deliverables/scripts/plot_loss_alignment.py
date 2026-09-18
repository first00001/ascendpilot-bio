#!/usr/bin/env python3
"""Render the formal AscendC/Triton loss-alignment evidence as SVG."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


ROW = re.compile(
    r"iteration\s+(?P<step>\d+)/\s*\d+.*?"
    r"elapsed time per iteration \(ms\):\s*(?P<ms>[0-9.]+).*?"
    r"global batch size:\s*(?P<gbs>\d+).*?loss:\s*(?P<loss>[0-9.Ee+-]+)"
)


def parse_log(path: Path) -> dict[int, float]:
    rows: dict[int, float] = {}
    for match in ROW.finditer(path.read_text(encoding="utf-8", errors="replace")):
        rows[int(match.group("step"))] = float(match.group("loss"))
    return rows


def points(values, x, y, width, height, x_min, x_max, y_min, y_max):
    def px(step):
        return x + (step - x_min) / (x_max - x_min) * width

    def py(value):
        return y + height - (value - y_min) / (y_max - y_min) * height

    return " ".join(f"{px(step):.2f},{py(value):.2f}" for step, value in values)


def render(ascendc_path: Path, triton_path: Path, output: Path):
    ascendc = parse_log(ascendc_path)
    triton = parse_log(triton_path)
    steps = sorted(set(ascendc) & set(triton))
    if not steps:
        raise SystemExit("no common training iterations")

    a_values = [(step, ascendc[step]) for step in steps]
    t_values = [(step, triton[step]) for step in steps]
    errors = [(step, abs(ascendc[step] - triton[step])) for step in steps]
    relative = [
        (step, error / max(abs(triton[step]), 1e-12) * 100)
        for (step, error) in errors
    ]

    mean_abs = sum(value for _, value in errors) / len(errors)
    max_step, max_abs = max(errors, key=lambda item: item[1])
    last_abs = errors[-1][1]
    mean_relative = sum(value for _, value in relative) / len(relative)
    max_relative = max(value for _, value in relative)

    width, height = 1600, 1120
    left, plot_width = 125, 1360
    top_y, panel_height = 155, 330
    bottom_y = 650
    all_losses = [value for _, value in a_values + t_values]
    loss_min = min(all_losses)
    loss_max = max(all_losses)
    loss_pad = max((loss_max - loss_min) * 0.12, 0.02)
    loss_min -= loss_pad
    loss_max += loss_pad
    error_max = max_abs * 1.18 if max_abs else 1.0

    def line(x1, y1, x2, y2, color="#d9e1e8", stroke=1, dash=""):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{color}" stroke-width="{stroke}"{extra}/>'
        )

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f7f9fb"/>',
        '<style>text{font-family:Arial,"Microsoft YaHei",sans-serif;fill:#17212b}'
        '.title{font-size:34px;font-weight:700}.subtitle{font-size:18px;fill:#52616f}'
        '.panel{font-size:24px;font-weight:700}.axis{font-size:16px;fill:#52616f}'
        '.metric{font-size:17px;fill:#293845}.legend{font-size:17px;font-weight:700}</style>',
        '<text x="800" y="54" text-anchor="middle" class="title">Qwen3.5-0.8B Loss Alignment</text>',
        '<text x="800" y="88" text-anchor="middle" class="subtitle">Formal 100-step dual-Ascend run · AscendC vs Triton · identical training configuration</text>',
        f'<rect x="80" y="115" width="1440" height="430" rx="6" fill="#fff" stroke="#cbd5df"/>',
        f'<rect x="80" y="610" width="1440" height="430" rx="6" fill="#fff" stroke="#cbd5df"/>',
        '<text x="125" y="145" class="panel">Training loss by iteration</text>',
        '<text x="125" y="640" class="panel">Per-step absolute loss difference</text>',
    ]

    for index in range(6):
        ratio = index / 5
        y = top_y + panel_height - ratio * panel_height
        value = loss_min + ratio * (loss_max - loss_min)
        svg.append(line(left, y, left + plot_width, y))
        svg.append(f'<text x="108" y="{y + 6:.1f}" text-anchor="end" class="axis">{value:.3f}</text>')

        error_y = bottom_y + panel_height - ratio * panel_height
        error_value = ratio * error_max
        svg.append(line(left, error_y, left + plot_width, error_y))
        svg.append(f'<text x="108" y="{error_y + 6:.1f}" text-anchor="end" class="axis">{error_value:.4f}</text>')

    tick_steps = [1, 20, 40, 60, 80, 100]
    for step in tick_steps:
        x = left + (step - steps[0]) / (steps[-1] - steps[0]) * plot_width
        svg.append(line(x, top_y, x, top_y + panel_height, "#edf1f4"))
        svg.append(line(x, bottom_y, x, bottom_y + panel_height, "#edf1f4"))
        svg.append(f'<text x="{x:.1f}" y="510" text-anchor="middle" class="axis">{step}</text>')
        svg.append(f'<text x="{x:.1f}" y="1005" text-anchor="middle" class="axis">{step}</text>')

    svg.extend(
        [
            line(left, top_y, left, top_y + panel_height, "#657786", 1.5),
            line(left, top_y + panel_height, left + plot_width, top_y + panel_height, "#657786", 1.5),
            line(left, bottom_y, left, bottom_y + panel_height, "#657786", 1.5),
            line(left, bottom_y + panel_height, left + plot_width, bottom_y + panel_height, "#657786", 1.5),
            f'<polyline points="{points(a_values, left, top_y, plot_width, panel_height, steps[0], steps[-1], loss_min, loss_max)}" fill="none" stroke="#176b4d" stroke-width="4" stroke-linejoin="round"/>',
            f'<polyline points="{points(t_values, left, top_y, plot_width, panel_height, steps[0], steps[-1], loss_min, loss_max)}" fill="none" stroke="#c4553d" stroke-width="3" stroke-linejoin="round"/>',
            f'<polyline points="{points(errors, left, bottom_y, plot_width, panel_height, steps[0], steps[-1], 0, error_max)}" fill="none" stroke="#326a8b" stroke-width="3" stroke-linejoin="round"/>',
            '<line x1="1120" y1="135" x2="1165" y2="135" stroke="#176b4d" stroke-width="4"/>',
            '<text x="1178" y="141" class="legend">AscendC</text>',
            '<line x1="1305" y1="135" x2="1350" y2="135" stroke="#c4553d" stroke-width="4"/>',
            '<text x="1363" y="141" class="legend">Triton</text>',
            '<text x="805" y="535" text-anchor="middle" class="axis">Training iteration</text>',
            '<text x="805" y="1030" text-anchor="middle" class="axis">Training iteration</text>',
            '<text x="35" y="320" text-anchor="middle" class="axis" transform="rotate(-90 35 320)">Loss</text>',
            '<text x="35" y="815" text-anchor="middle" class="axis" transform="rotate(-90 35 815)">Absolute difference</text>',
            f'<text x="125" y="1080" class="metric">Mean abs diff: {mean_abs:.6f}   ·   Max abs diff: {max_abs:.6f} (step {max_step})   ·   Step 100 diff: {last_abs:.6f}</text>',
            f'<text x="1485" y="1080" text-anchor="end" class="metric">Mean relative abs: {mean_relative:.4f}%   ·   Max relative abs: {max_relative:.4f}%</text>',
            '<text x="800" y="1105" text-anchor="middle" class="axis">No acceptance threshold is drawn; final precision acceptance follows the competition evaluator.</text>',
            '</svg>',
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(svg), encoding="utf-8")
    print(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ascendc", type=Path, required=True)
    parser.add_argument("--triton", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    render(args.ascendc, args.triton, args.output)


if __name__ == "__main__":
    main()
