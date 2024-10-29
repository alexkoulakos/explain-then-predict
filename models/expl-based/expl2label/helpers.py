import torch
import random
import numpy as np
import numpy as np

from logging import Logger

from torch.distributed import init_process_group, destroy_process_group

def set_seed(args) -> None:
    """
    Set the random seed across all devices for reproducibility.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    if args.n_gpus > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup(args) -> None:
    """
    Initiate nccl backend for distributed training. 
    """
    init_process_group(backend="nccl")
    torch.cuda.set_device(args.local_rank)

def cleanup() -> None:
    """
    Kill existing processes that were used in the distributed training. 
    """
    destroy_process_group()

def compute_time(start_time, end_time) -> dict:
    """
    Calculate elapsed time. Patricularly useful for the calculation of total training time.
    Returns a `dict` with the elapsed hours and minutes.
    """
    elapsed_time = end_time - start_time
    elapsed_hours = int(elapsed_time / 3600)
    elapsed_mins = int((elapsed_time - (elapsed_hours * 3600)) / 60)

    return {
        'elapsed_hours': elapsed_hours,
        'elapsed_mins': elapsed_mins
    }

def display_training_progress(
        logger: Logger,
        step, 
        progress, 
        loss, 
        acc, 
        elapsed_hours, 
        elapsed_mins, 
        expl, 
        gold_label, 
        pred_label
    ) -> None:
    """
    Display training info in order to be able to monitor the training process. 
    """
    logger.info(f"iter: {step} | progress: {progress:.2f}%")
    logger.info(f"avg. train loss: {loss} | avg. train acc: {acc} | time elapsed: {elapsed_hours} hours {elapsed_mins} mins")
    logger.info(f"EXPLANATION: {expl}")
    logger.info(f"GOLD LABEL: {gold_label}")
    logger.info(f"PREDICTED LABEL: {pred_label}\n")