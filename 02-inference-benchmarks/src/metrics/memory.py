"""GPU memory profiling via nvidia-smi."""

import subprocess


def get_gpu_memory() -> tuple[int, int]:
    """Returns (used_mb, total_mb) from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return parse_gpu_memory(result.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return 0, 0


def parse_gpu_memory(output: str) -> tuple[int, int]:
    """Parse nvidia-smi memory output like '45000 MiB, 81920 MiB' or '45000, 81920'."""
    parts = output.replace("MiB", "").split(",")
    used = int(parts[0].strip())
    total = int(parts[1].strip())
    return used, total
