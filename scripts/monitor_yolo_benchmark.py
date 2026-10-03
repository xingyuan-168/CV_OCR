"""Run a benchmark while recording optional NVIDIA telemetry to a separate CSV."""
import argparse
from pathlib import Path
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("Supply -- followed by the benchmark executable and its arguments")
    smi = shutil.which("nvidia-smi")
    monitor = None
    with args.gpu_log.open("wb") as file:
        if smi:
            monitor = subprocess.Popen([smi, "--query-gpu=timestamp,index,name,driver_version,utilization.gpu,utilization.memory,memory.used,memory.total", "--format=csv", "-l", "1"], stdout=file, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            result = subprocess.run(command)
        finally:
            if monitor:
                monitor.terminate()
                monitor.wait()
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
