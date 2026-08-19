"""Lesson 28: analyze position, confusion, and confidence changes between two CNNs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True, slots=True)
class ComparisonRow:
    filename: str
    target: str
    first: str
    second: str
    first_confidences: tuple[float, ...]
    second_confidences: tuple[float, ...]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comparison_csv",
        type=Path,
        nargs="?",
        default=Path("artifacts/checkpoint_comparison/comparison.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/checkpoint_change_analysis"),
    )
    return parser


def load_rows(path: Path) -> list[ComparisonRow]:
    with path.expanduser().resolve().open(newline="", encoding="utf-8-sig") as stream:
        raw_rows = list(csv.DictReader(stream))
    if not raw_rows:
        raise ValueError("comparison CSV is empty")

    rows: list[ComparisonRow] = []
    for raw in raw_rows:
        target = raw["target"]
        first = raw["first_prediction"]
        second = raw["second_prediction"]
        if not (len(target) == len(first) == len(second) == 4):
            raise ValueError(f"expected four-character texts in row: {raw['filename']}")
        try:
            first_confidences = tuple(
                float(raw[f"first_confidence_{position}"]) for position in range(1, 5)
            )
            second_confidences = tuple(
                float(raw[f"second_confidence_{position}"]) for position in range(1, 5)
            )
        except KeyError as error:
            raise ValueError(
                "comparison CSV has no per-position confidence columns; rerun lesson 27 first"
            ) from error
        rows.append(
            ComparisonRow(
                filename=raw["filename"],
                target=target,
                first=first,
                second=second,
                first_confidences=first_confidences,
                second_confidences=second_confidences,
            )
        )
    return rows


def position_statistics(rows: list[ComparisonRow]) -> list[tuple[int, int, int, int, int]]:
    statistics: list[tuple[int, int, int, int, int]] = []
    for position in range(4):
        first_correct = sum(row.first[position] == row.target[position] for row in rows)
        second_correct = sum(row.second[position] == row.target[position] for row in rows)
        fixed = sum(
            row.first[position] != row.target[position]
            and row.second[position] == row.target[position]
            for row in rows
        )
        regressed = sum(
            row.first[position] == row.target[position]
            and row.second[position] != row.target[position]
            for row in rows
        )
        statistics.append((position + 1, first_correct, second_correct, fixed, regressed))
    return statistics


def character_statistics(
    rows: list[ComparisonRow],
) -> list[tuple[str, int, int, int, int]]:
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        for expected, first, second in zip(row.target, row.first, row.second):
            counts[expected][0] += 1
            counts[expected][1] += first == expected
            counts[expected][2] += second == expected
    return sorted(
        (
            (character, total, first_correct, second_correct, second_correct - first_correct)
            for character, (total, first_correct, second_correct) in counts.items()
        ),
        key=lambda item: (item[4], item[1], item[0]),
    )


def build_report(rows: list[ComparisonRow]) -> str:
    position_stats = position_statistics(rows)
    character_stats = character_statistics(rows)
    first_confusions: Counter[tuple[str, str]] = Counter()
    second_confusions: Counter[tuple[str, str]] = Counter()
    fixed_events: Counter[tuple[str, str]] = Counter()
    regressed_events: Counter[tuple[str, str]] = Counter()
    confident_second_errors: list[tuple[float, str, int, str, str]] = []

    for row in rows:
        for position, (expected, first, second) in enumerate(
            zip(row.target, row.first, row.second), start=1
        ):
            if first != expected:
                first_confusions[(expected, first)] += 1
            if second != expected:
                second_confusions[(expected, second)] += 1
                confident_second_errors.append(
                    (
                        row.second_confidences[position - 1],
                        row.filename,
                        position,
                        expected,
                        second,
                    )
                )
            if first != expected and second == expected:
                fixed_events[(expected, first)] += 1
            if first == expected and second != expected:
                regressed_events[(expected, second)] += 1

    lines = [
        f"Images: {len(rows)}",
        "",
        "Position accuracy and transitions:",
    ]
    for position, first_correct, second_correct, fixed, regressed in position_stats:
        lines.append(
            f"  position {position}: v1={first_correct}/{len(rows)}={first_correct / len(rows):.2%}, "
            f"v2={second_correct}/{len(rows)}={second_correct / len(rows):.2%}, "
            f"delta={second_correct - first_correct:+d}, fixed={fixed}, regressed={regressed}"
        )

    lines.extend(["", "Character accuracy (sorted from worst change to best):"])
    for character, total, first_correct, second_correct, delta in character_stats:
        lines.append(
            f"  {character}: samples={total}, v1={first_correct}/{total}, "
            f"v2={second_correct}/{total}, delta={delta:+d}"
        )

    lines.extend(["", "Character slots fixed by v2 (target <- v1 prediction):"])
    lines.extend(
        f"  {target} <- {wrong}: {count}"
        for (target, wrong), count in fixed_events.most_common()
    )
    lines.extend(["", "Character slots regressed in v2 (target -> v2 prediction):"])
    lines.extend(
        f"  {target} -> {wrong}: {count}"
        for (target, wrong), count in regressed_events.most_common()
    )

    lines.extend(["", "Top v1 confusions:"])
    lines.extend(
        f"  {target} -> {wrong}: {count}"
        for (target, wrong), count in first_confusions.most_common(12)
    )
    lines.extend(["", "Top v2 confusions:"])
    lines.extend(
        f"  {target} -> {wrong}: {count}"
        for (target, wrong), count in second_confusions.most_common(12)
    )

    lines.extend(["", "Highest-confidence v2 errors:"])
    for confidence, filename, position, target, prediction in sorted(
        confident_second_errors, reverse=True
    )[:12]:
        lines.append(
            f"  {filename} position={position}: {target} -> {prediction}, confidence={confidence:.2%}"
        )
    return "\n".join(lines)


def save_position_chart(rows: list[ComparisonRow], path: Path) -> None:
    statistics = position_statistics(rows)
    width, height = 720, 420
    chart = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(chart)
    font = ImageFont.load_default()
    left, top, bottom = 70, 40, 350
    plot_height = bottom - top

    draw.line((left, top, left, bottom), fill="black", width=2)
    draw.line((left, bottom, width - 30, bottom), fill="black", width=2)
    for percentage in range(0, 101, 20):
        y = bottom - round(plot_height * percentage / 100)
        draw.line((left, y, width - 30, y), fill="#dddddd")
        draw.text((28, y - 6), f"{percentage}%", fill="black", font=font)

    for index, (position, first_correct, second_correct, _, _) in enumerate(statistics):
        group_x = left + 70 + index * 145
        for offset, correct, color, name in (
            (-25, first_correct, "#777777", "v1"),
            (25, second_correct, "#2878b5", "v2"),
        ):
            accuracy = correct / len(rows)
            bar_height = round(plot_height * accuracy)
            x = group_x + offset
            draw.rectangle((x - 18, bottom - bar_height, x + 18, bottom), fill=color)
            draw.text((x - 18, bottom - bar_height - 16), f"{accuracy:.0%}", fill=color, font=font)
            draw.text((x - 7, bottom + 6), name, fill=color, font=font)
        draw.text((group_x - 27, bottom + 25), f"position {position}", fill="black", font=font)

    draw.text((left, 15), "Real captcha accuracy by output position", fill="black", font=font)
    chart.save(path)


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows(args.comparison_csv)
    output_directory = args.output.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    report_path = output_directory / "analysis.txt"
    chart_path = output_directory / "position_accuracy.png"
    report = build_report(rows)
    report_path.write_text(report + "\n", encoding="utf-8")
    save_position_chart(rows, chart_path)
    print(report)
    print(f"\nReport: {report_path}")
    print(f"Position chart: {chart_path}")


if __name__ == "__main__":
    main()
