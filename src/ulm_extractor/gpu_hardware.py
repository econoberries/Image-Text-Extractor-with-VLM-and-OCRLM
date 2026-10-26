"""Query the GPU actually assigned to this Slurm worker for provenance."""
import subprocess


def gpu_hardware():
    response = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    lines = [line.strip() for line in response.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Expected a single allocated GPU: {lines}")
    name, total = (part.strip() for part in lines[0].rsplit(",", 1))
    if not name or not total.isdigit():
        raise ValueError(f"Cannot parse GPU hardware: {lines[0]}")
    return {"name": name, "memory_mib": int(total)}
