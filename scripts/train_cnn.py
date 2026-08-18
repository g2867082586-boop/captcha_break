"""Train the fixed-length CNN; additional CLI arguments are forwarded."""

import sys

from captcha_break.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", "--model", "cnn", *sys.argv[1:]]))
