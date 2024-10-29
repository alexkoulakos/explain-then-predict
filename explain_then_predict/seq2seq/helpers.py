import torch
import random
import numpy as np
import evaluate
import re
import numpy as np
import pandas as pd

from typing import List
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

def shift_tokens_left(input_ids: torch.tensor) -> torch.tensor: 
    """
    Shift tokens represented by `input_ids` to the left and insert -100 at the end.
    This is essential for accurate loss calculation inside the `EncoderDecoderModel` `forward()` method.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, :-1] = input_ids[:, 1:].clone()
    
    shifted_input_ids[:, -1] = -100

    return shifted_input_ids

def truncate_pred_expls(pred_expls: List[str]) -> List[str]:
    return list(map(lambda x: x.split('.')[0] + '.', pred_expls)) 

def remove_punctuation(sentences: List[str]) -> List[str]:
    """
    Remove punctuation from each sentence in `sentences`.
    This is a preliminary step for the accurate calculation of evaluation metrics (METEOR, BLEU, ROUGE, BERTScore).
    Returns a `list` with the non-punctuated sentences.
    """
    return list(map((lambda x: re.sub(r'[^\w\s]', '', x)), sentences))

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

def preprocess(pred_file, args) -> dict:
    """
    Remove punctuation from predicted and ground truth explanations contained in `pred_file`.
    This is a preliminary step for the accurate calculation of evaluation metrics (METEOR, BLEU, ROUGE, BERTScore).
    Returns a `dict` with the non-punctuated sentences.
    """
    df = pd.read_csv(pred_file)

    predictions = df['pred_explanation']
    expls_1 = df['explanation_1']
    expls_2 = df['explanation_2']
    expls_3 = df['explanation_3']
    
    if args.remove_punctuation:
        predictions = remove_punctuation(predictions)
        expls_1 = remove_punctuation(expls_1)
        expls_2 = remove_punctuation(expls_2)
        expls_3 = remove_punctuation(expls_3)

    references = [[expl_1, expl_2, expl_3] for expl_1, expl_2, expl_3 in zip(expls_1, expls_2, expls_3)]

    return {
        'predictions': predictions,
        'references': references
    }

def compute_metrics(predictions: List[str], references: List[List[str]]) -> dict:
    """
    Compute METEOR, BLEU, ROUGE-1, ROUGE-2 and BERTScore F1 metrics.
    Returns a `dict` with the calculated metrics.
    """
    bleu = evaluate.load('bleu')
    meteor = evaluate.load('meteor')
    rouge = evaluate.load('rouge')
    bert = evaluate.load('bertscore')

    bleu_score = bleu.compute(predictions=predictions, references=references)['bleu']
    meteor_score = meteor.compute(predictions=predictions, references=references)['meteor']
    rouge_scores = rouge.compute(predictions=predictions, references=references, rouge_types=['rouge1', 'rouge2'], use_aggregator=True)
    rouge_1_score, rouge_2_score = rouge_scores['rouge1'], rouge_scores['rouge2']
    bert_score = np.mean(bert.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")['f1'])

    return {
        'bleu_score': bleu_score,
        'meteor_score': meteor_score,
        'rouge_1_score': rouge_1_score,
        'rouge_2_score': rouge_2_score,
        'bert_score': bert_score
    }

def display_training_progress(
        logger: Logger, 
        step, 
        progress, 
        loss, 
        ppl, 
        elapsed_hours, 
        elapsed_mins, 
        premise, 
        hypothesis, 
        expl, 
        decoded_expl
    ) -> None:
    """
    Display training info in order to be able to monitor the training process. 
    """
    logger.info(f"iter: {step} | progress: {progress:.2f}%")
    logger.info(f"avg. train loss: {loss} | avg. train ppl: {ppl} | time elapsed: {elapsed_hours} hours {elapsed_mins} mins")
    logger.info(f"PREMISE: {premise}")
    logger.info(f"HYPOTHESIS: {hypothesis}")
    logger.info(f"EXPLANATION: {expl}")
    logger.info(f"DECODED EXPLANATION: {decoded_expl}\n")

def new_best_model_found(current_score, best_score_so_far, validation_metric) -> bool:
    """
    Checks whether a new best model has been found according to the specified `validation_metric`.
    If the validation metric is perplexity, a decrease in perplexity indicates that a new best model has been found.
    Otherwise, an increase indicates that a new best model has been found.
    Returns `True` if a new best model has been found, `False` otherwise.
    """
    if validation_metric == 'ppl' and current_score < best_score_so_far:
        return True
    
    if validation_metric != 'ppl' and current_score > best_score_so_far:
        return True
    
    return False