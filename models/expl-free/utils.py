import torch
import random
import numpy as np
import numpy as np

from typing import Dict
from logging import Logger

from torch.distributed import init_process_group, destroy_process_group

ID2LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}

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

def compute_time(start_time: int, end_time: int) -> Dict:
    """
    Calculate elapsed time. Patricularly useful for the calculation of total training time.
    Returns a `Dict` with the elapsed minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return {
        'elapsed_mins': elapsed_mins,
        'elapsed_secs': elapsed_secs
    }

def display_training_progress(
        logger: Logger, 
        device: str, 
        step: int, 
        progress: str, 
        loss: str, 
        acc: str, 
        mins_elapsed: int, 
        secs_elapsed: int, 
        premise: str, 
        hypothesis: str, 
        gold_label: str, 
        pred_label: str, 
        mode: str
    ) -> None:
    """
    Display training info in order to be able to monitor the training process. 
    """
    assert mode in ['train', 'eval']

    logger.info(f"Device: {device}")
    logger.info(f"iter: {step} | progress: {progress:.2f}%")
    logger.info(f"avg. {mode} loss: {loss} | avg. {mode} acc: {acc} | time elapsed: {mins_elapsed} mins {secs_elapsed} secs")
    logger.info(f"PREMISE: {premise}")
    logger.info(f"HYPOTHESIS: {hypothesis}")
    logger.info(f"GOLD LABEL: {gold_label}")
    logger.info(f"PREDICTED LABEL: {pred_label}")