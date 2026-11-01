"""Does a user-triggered GPU core dump name a kernel that is still running? The positive control for run.sh's dump of a
stalled engine.

A child process launches torch's spin kernel (torch.cuda._sleep) for about 20 s with CUDA_ENABLE_USER_TRIGGERED_COREDUMP
on; while it spins, this writes to the child's pipe, waits for the dump to stop growing, lets the kernel finish, and
reads the dump with cuda-gdb (target cudacore, info cuda kernels). It passes if cuda-gdb names the spin kernel.

usage: python dump_selftest.py DUMP_DIR CUDA_GDB OUT.json   (needs the GPU to itself)
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CYCLES = 50_000_000_000  # about 20 s at the card's 2.5-3 GHz
FLAGS = "skip_nonrelocated_elf_images,skip_global_memory,skip_shared_memory,skip_local_memory,skip_constbank_memory"
CHILD = f"""
import torch
torch.zeros(1, device="cuda")
torch.cuda.synchronize()
torch.cuda._sleep({CYCLES})
print("launched", flush=True)
torch.cuda.synchronize()
print("finished", flush=True)
"""


def dump_size(files):
    return sum(f.stat().st_size for f in files if f.is_file())


def main():
    dumps, cuda_gdb, out = Path(sys.argv[1]).resolve(), sys.argv[2], sys.argv[3]
    dumps.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CUDA_ENABLE_USER_TRIGGERED_COREDUMP="1",
        CUDA_COREDUMP_PIPE=str(dumps / "selftest-pipe.%p"),
        CUDA_COREDUMP_FILE=str(dumps / "selftest.%p.%t"),
        CUDA_COREDUMP_GENERATION_FLAGS=FLAGS,
        CUDA_COREDUMP_SHOW_PROGRESS="1",
    )
    result = {"cycles": CYCLES, "flags": FLAGS}
    child = subprocess.Popen(
        [sys.executable, "-c", CHILD], env=env, stdout=subprocess.PIPE, text=True
    )
    assert child.stdout.readline().strip() == "launched", "the child did not launch its kernel"
    time.sleep(2)
    pipe = dumps / f"selftest-pipe.{child.pid}"
    result["pipe_exists"] = pipe.is_fifo()
    began = time.monotonic()
    try:
        # Fails at once (ENXIO) if the driver is not reading the pipe.
        fd = os.open(pipe, os.O_WRONLY | os.O_NONBLOCK)
        os.write(fd, b"\n")
        os.close(fd)
        result["requested"] = True
    except OSError as e:
        result["requested"] = f"no: {e}"
    if result["requested"] is True:
        last, same = -1, 0
        while time.monotonic() - began < 120 and same < 5:
            size = dump_size(dumps.glob(f"selftest.{child.pid}.*"))
            same = same + 1 if size > 0 and size == last else 0
            last = size
            time.sleep(1)
    result["dump_s"] = round(time.monotonic() - began, 1)
    files = sorted(dumps.glob(f"selftest.{child.pid}.*"))
    result["dump_files"] = [f.name for f in files]
    result["dump_bytes"] = dump_size(files)
    try:  # the kernel runs out its cycles and the child exits, unless the dump left the GPU halted
        result["child_exit"] = child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        child.kill()
        result["child_exit"] = "killed after 120 s"
    # "finished" once the kernel ran out its cycles.
    result["child_after"] = child.stdout.read().strip()
    result["kernel"] = None
    if files:
        gdb = subprocess.run(
            [
                cuda_gdb,
                "-batch",
                "-nx",
                "-ex",
                f"target cudacore {files[0]}",
                "-ex",
                "info cuda kernels",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        result["cuda_gdb_exit"] = gdb.returncode
        result["cuda_gdb"] = (gdb.stdout + gdb.stderr)[-3000:]
        result["kernel"] = next(
            (line.strip() for line in gdb.stdout.splitlines() if "spin_kernel" in line), None
        )
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps({k: v for k, v in result.items() if k != "cuda_gdb"}, indent=1))
    sys.exit(0 if result["kernel"] else 1)


if __name__ == "__main__":
    main()
