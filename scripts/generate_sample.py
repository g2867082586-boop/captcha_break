"""Generate a captcha from a regular Python script."""

import sys

from captcha_break.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["sample", *sys.argv[1:]]))
