import torch
import pandas as pd
import numpy as np
import random
import re
import evaluate

from typing import List

def set_seed(args) -> None:
    """
    Set the random seed for reproducibility.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

def remove_punctuation(sentences: List[str]) -> List[str]:
    """
    Remove punctuation from each sentence in `sentences`.
    This is a preliminary step for the accurate calculation of evaluation metrics (METEOR, BLEU, ROUGE, BERTScore).
    Returns a `list` with the non-punctuated sentences.
    """
    return list(map((lambda x: re.sub(r'[^\w\s]', '', x)), sentences))

def preprocess(pred_file, args) -> dict:
    """
    Remove punctuation from predicted and ground truth explanations contained in `pred_file`.
    This is a preliminary step for the accurate calculation of NLG evaluation metrics (METEOR, BLEU, ROUGE, BERTScore).
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

def compute_nlg_metrics(predictions: List[str], references: List[List[str]]) -> dict:
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

def convert_labels_to_ids(labels, label2id) -> torch.tensor:
    return torch.tensor(list(map(lambda x: label2id[x], labels))).to('cpu')

def compute_acc(
        pred_file,
        id2label: dict,
        label2id: dict
    ) -> dict:
    df = pd.read_csv(pred_file)

    pred_label_ids = convert_labels_to_ids(df['pred_label'], label2id)
    gold_label_ids = convert_labels_to_ids(df['gold_label'], label2id)

    accuracy = {}

    for label in label2id.keys():
        accuracy[label] = 0
    
    for pred_label_id, gold_label_id in zip(pred_label_ids.tolist(), gold_label_ids.tolist()):
        gold_label = id2label[gold_label_id]

        if pred_label_id == gold_label_id:
            accuracy[gold_label] += 1
    
    for label in accuracy.keys():
        accuracy[label] /= gold_label_ids.tolist().count(label2id[label])

    acc = torch.sum(pred_label_ids==gold_label_ids) / len(pred_label_ids)

    accuracy['total'] = acc.item()

    return accuracy