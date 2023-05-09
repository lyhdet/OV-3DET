# Copyright (c) Facebook, Inc. and its affiliates.

import torch
import os
from utils.dist import is_primary


def save_checkpoint(
    checkpoint_dir,
    model_no_ddp,
    optimizer,
    epoch,
    args,
    best_val_metrics,
    filename=None,
):
    if not is_primary():
        return
    if filename is None:
        filename = f"checkpoint_{epoch:04d}.pth"
    checkpoint_name = os.path.join(checkpoint_dir, filename)

    sd = {
        "model": model_no_ddp.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "args": args,
        "best_val_metrics": best_val_metrics,
    }
    torch.save(sd, checkpoint_name)


def resume_if_possible(checkpoint_dir, model_no_ddp, optimizer):
    """
    Resume if checkpoint is available.
    Return
    - epoch of loaded checkpoint.
    """
    epoch = -1
    best_val_metrics = {}
    if not os.path.isdir(checkpoint_dir):
        return epoch, best_val_metrics

    last_checkpoint = os.path.join(checkpoint_dir, "checkpoint.pth")
    if not os.path.isfile(last_checkpoint):
        return epoch, best_val_metrics

    sd = torch.load(last_checkpoint, map_location=torch.device("cpu"))
    epoch = sd["epoch"]
    best_val_metrics = sd["best_val_metrics"]
    print(f"Found checkpoint at {epoch}. Resuming.")

    resume_var = {}

    for key,val in sd["model"].items():
        #if "img_model" in key or "clip_header" in key or "sem_cls_head" in key:
        # if "img_model" in key or "pc_model.mlp_heads.sem_cls_head" in key:
        if "img_model" in key:
        #if "img_model" in key or "pc_model.clip_header" in key:
            continue
        resume_var[key]=val

    for key,val in resume_var.items():
        print(key)

    #input()

    #model_no_ddp.load_state_dict(sd["model"])
    model_no_ddp.load_state_dict(resume_var, strict=False)
    #optimizer.load_state_dict(sd["optimizer"])
    print(
        f"Loaded model and optimizer state at {epoch}. Loaded best val metrics so far."
    )
    # return epoch, best_val_metrics
    return -1, {}
    
