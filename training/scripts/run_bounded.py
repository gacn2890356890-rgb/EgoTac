"""Finite parent watchdog for one local training process; no job monitoring loop."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=int, default=14400)
    parser.add_argument("--report", help="Local report path, written after child exits")
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 1 <= args.seconds <= 14400:
        parser.error("Wall budget must be between 1 and 14400 seconds")
    command_args = args.training_args
    if command_args and command_args[0] == "--":
        command_args = command_args[1:]
    if not command_args:
        parser.error("Pass training arguments after --")
    stamp = time.monotonic()
    process = subprocess.Popen([sys.executable, "-m", "egotouch.train", *command_args], stdin=subprocess.DEVNULL)
    state = "exited"
    try:
        code = process.wait(timeout=args.seconds)
    except subprocess.TimeoutExpired:
        state = "wall_timeout"
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        code = 124
    except KeyboardInterrupt:
        state = "interrupted"
        process.terminate()
        process.wait()
        code = 130
    result = {"state": state, "pid": process.pid, "exit_code": code, "elapsed_s": time.monotonic() - stamp}
    if args.report:
        target = Path(args.report)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
