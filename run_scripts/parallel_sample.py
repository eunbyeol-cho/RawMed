"""
Parallel AR sampling across multiple GPUs.

Launches independent processes per GPU for maximum throughput,
then concatenates results.

Usage:
    python run_scripts/parallel_sample.py \
        gpu_ids=0,1,2,3 gen_samples=5000 output_dir=/path/to/output \
        ehr=eicu max_event_size=114 ...
"""
import os
import sys
import glob
import shutil
import subprocess
import numpy as np


def main():
    # Parse all args as key=value
    own_keys = {"gpu_ids", "gen_samples", "output_dir"}
    own_args = {}
    sacred_args = []

    for arg in sys.argv[1:]:
        if "=" in arg:
            key, val = arg.split("=", 1)
            if key in own_keys:
                own_args[key] = val
            else:
                sacred_args.append(arg)
        else:
            sacred_args.append(arg)

    for key in own_keys:
        if key not in own_args:
            print(f"ERROR: missing required argument: {key}=...")
            sys.exit(1)

    gpus = own_args["gpu_ids"].split(",")
    gen_samples = int(own_args["gen_samples"])
    output_dir = own_args["output_dir"]
    num_gpus = len(gpus)
    per_gpu = gen_samples // num_gpus
    remainder = gen_samples % num_gpus
    tmp_dir = os.path.join(output_dir, ".tmp_parallel_gen")

    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)

    # Launch parallel processes
    procs = []
    for i, gpu in enumerate(gpus):
        samples = per_gpu + (remainder if i == num_gpus - 1 else 0)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["SEED_OFFSET"] = str(i)
        env["OMP_NUM_THREADS"] = "8"
        env["NUMEXPR_MAX_THREADS"] = "128"

        cmd = [
            sys.executable, "main.py", "with", "task_sample_AR",
            *sacred_args,
            f"generated_data_path={tmp_dir}/gpu{i}",
            f"gen_samples={samples}",
        ]
        print(f"  GPU {gpu}: {samples} samples (seed_offset={i})")
        procs.append(subprocess.Popen(cmd, env=env))

    # Wait for all
    failed = False
    for p in procs:
        if p.wait() != 0:
            print(f"ERROR: Process {p.pid} failed (exit code {p.returncode})")
            failed = True
    if failed:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        sys.exit(1)

    # Concatenate results
    gpu_dirs = sorted(glob.glob(os.path.join(tmp_dir, "gpu*")))
    subfolder = os.listdir(gpu_dirs[0])[0]
    final_dir = os.path.join(output_dir, subfolder)
    os.makedirs(final_dir, exist_ok=True)

    for npy_file in sorted(glob.glob(os.path.join(gpu_dirs[0], subfolder, "*.npy"))):
        fname = os.path.basename(npy_file)
        arrays = [np.load(os.path.join(d, subfolder, fname)) for d in gpu_dirs]
        result = np.concatenate(arrays, axis=0)
        np.save(os.path.join(final_dir, fname), result)
        print(f"  {fname}: {result.shape}")

    print(f"Saved to {final_dir}")
    shutil.rmtree(tmp_dir)


if __name__ == "__main__":
    main()
