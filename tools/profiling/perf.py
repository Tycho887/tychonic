#!/usr/bin/env python3
"""Attach perf after fixture creation/warm-up and enable it before measured work."""
import argparse
import os
from pathlib import Path
import select
import signal
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["stat", "record"], default="stat")
    parser.add_argument("--perf-binary", default="perf", help="perf executable or wrapper path")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events", default="cycles:u,instructions:u,branches:u,branch-misses:u,cache-misses:u")
    parser.add_argument("runner", nargs=argparse.REMAINDER, help="-- BINARY [runner options]")
    args = parser.parse_args()
    command = args.runner
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("supply -- BINARY [runner options]")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["SATKIT_OFFLINE"] = "1"
    runner = subprocess.Popen(command + ["--wait-for-start"], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    profiler = None
    descriptors = []
    try:
        while True:
            line = runner.stderr.readline()
            if line.startswith("READY pid="):
                break
            if not line:
                _, stderr = runner.communicate()
                raise RuntimeError(f"runner failed before READY: {stderr}")
            print(line, end="", flush=True)
        ctl_read, ctl_write = os.pipe()
        ack_read, ack_write = os.pipe()
        descriptors = [ctl_read, ctl_write, ack_read, ack_write]
        perf = [args.perf_binary, args.kind, "--pid", str(runner.pid), "--delay=-1",
                "--control", f"fd:{ctl_read},{ack_write}", "--output", str(args.output)]
        if args.kind == "stat":
            perf += ["--event", args.events, "-x", ","]
        else:
            perf += ["--event", "cpu-clock:u", "--freq", "499", "--call-graph", "dwarf", "--no-buildid-cache"]
        profiler = subprocess.Popen(perf, pass_fds=(ctl_read, ack_write), env=env)
        # The child now owns these ends; closing ours makes startup errors visible as EOF.
        os.close(ctl_read)
        os.close(ack_write)
        descriptors = [ctl_write, ack_read]
        os.write(ctl_write, b"enable\n")
        ready = select.select([ack_read], [], [], 10)[0]
        ack = os.read(ack_read, 4096) if ready else b""
        if not ack.startswith(b"ack\n"):
            raise RuntimeError(f"perf did not acknowledge enabling events ({ack!r}); inspect its diagnostic")
        runner.stdin.write("\n")
        runner.stdin.flush()
        stdout, stderr = runner.communicate()
        if runner.returncode:
            raise RuntimeError(f"runner failed: {stderr}")
        args.output.with_name(args.output.name + ".runner.json").write_text(stdout)
        try:
            profiler.wait(timeout=3)
        except subprocess.TimeoutExpired:
            profiler.send_signal(signal.SIGINT)
            profiler.wait(timeout=10)
        if profiler.returncode not in (0, -signal.SIGINT):
            raise RuntimeError(f"perf exited with {profiler.returncode}")
        print(stdout, end="", flush=True)
    finally:
        if runner.poll() is None:
            runner.terminate()
            runner.communicate()
        if profiler is not None and profiler.poll() is None:
            profiler.send_signal(signal.SIGINT)
            profiler.wait(timeout=10)
        for fd in descriptors:
            os.close(fd)


if __name__ == "__main__":
    main()
