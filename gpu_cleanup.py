import sys
import traceback
import torch
import gc

def format_peak_vram(device, *, label="Peak VRAM:"):
    device = torch.device(device)
    if device.type != "cuda":
        return f"{label} N/A ({device.type.upper()})"
    peak_mib = torch.cuda.max_memory_allocated(device) / 1024**2
    return f"{label} {peak_mib:.2f} MiB"


def release_gpu():
    """Release optimizer/module caches and CUDA workspaces after model disposal."""

    bnb = sys.modules.get("bitsandbytes")
    if bnb is not None and hasattr(bnb, "optim"):
        bnb.optim.GlobalOptimManager.get_instance().initialize()
    gc.collect()
    gc.collect()
    if torch.cuda.is_initialized():
        torch.cuda.synchronize()
        clear_workspaces = getattr(getattr(torch, "_C", None), "_cuda_clearCublasWorkspaces", None)
        if clear_workspaces is not None:
            clear_workspaces()
        torch.cuda.empty_cache()

def free_gpu_resources():
    """Call in a new cell after an interrupted run has stopped."""

    # Drop this notebook's live models/training objects; keep data and checkpoints.
    for name in ("trainer", "model", "base_model", "optimizer", "scheduler"):
        globals().pop(name, None)

    # An interrupted Trainer/generate call can survive through traceback locals.
    for name in ("last_traceback", "last_value", "last_exc"):
        value = getattr(sys, name, None)
        tb = value if name == "last_traceback" else getattr(value, "__traceback__", None)
        if tb is not None:
            traceback.clear_frames(tb)
        if hasattr(sys, name):
            setattr(sys, name, None)
    shell = globals().get("get_ipython", lambda: None)()
    if shell is not None:
        shell.run_line_magic("reset", "-f out")  # Clear cached Python results.
        shell.last_execution_result = None
        tb = getattr(shell.InteractiveTB, "tb", None)
        if tb is not None:
            traceback.clear_frames(tb)
            shell.InteractiveTB.tb = None
    try:
        release_gpu()
        if not torch.cuda.is_initialized():
            print("CUDA is not initialized; Python GPU references cleared.")
            return
        print(f"GPU cleanup: allocated {torch.cuda.memory_allocated()/1024**3:.2f} GiB; "
              f"reserved {torch.cuda.memory_reserved()/1024**3:.2f} GiB.")
    except RuntimeError as error:
        print(f"CUDA cleanup failed: {error}. Restart the kernel before loading another model.")


#free_gpu_resources()
