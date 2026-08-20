"""Lesson 36: call the reusable project recognizer for one local image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from captcha_break.recognizer import ProjectCaptchaRecognizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="path to one local CAPTCHA image")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    # Create this object once when your application starts, then reuse it.
    recognizer = ProjectCaptchaRecognizer()
    result = recognizer.recognize(args.image)

    print(json.dumps(result.as_dict(), ensure_ascii=False, indent=2))
    print(f"\nRecognized text: {result.text}")
    if not result.is_structurally_valid:
        print("Warning: neither model produced a structurally valid result.")


if __name__ == "__main__":
    main()
