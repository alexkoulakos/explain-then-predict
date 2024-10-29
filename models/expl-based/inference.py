import torch
import transformers
import argparse
import os
import csv
import logging
import json

from torch.utils.data import DataLoader, SequentialSampler
from transformers import AutoTokenizer, GenerationConfig

from model import ExplainThenPredictModel
from utils import *
from dataset import EsnliDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--seq2seq_model", type=str, required=True, 
                        help="Path to the fine-tuned encoder-decoder model directory.")
    parser.add_argument("--expl2label_model", type=str, required=True, 
                        help="Path to the fine-tuned classifier model file.")
    
    parser.add_argument("--generation_config_dir", required=True, type=str, 
                        help="Path to directory that contains the generatoon config files for model evaluation.")
    
    parser.add_argument("--test_data", default=None, type=str, required=True, 
                        help="Path to test data csv file.")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files.")

    # Non-required params
    parser.add_argument("--encoder_max_len", type=int, default=128, 
                        help="Max number of tokens the seq2seq encoder can process.")
    
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for testing.")
    parser.add_argument("--num_samples", type=int, default=-1, 
                        help="Number of samples to use for testing (-1 corresponds to the entire test set).")
    
    parser.add_argument("--remove_punctuation", action="store_true",
                        help="Whether to remove punctuation when calculating NLG evaluation metrics.")
    
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for initialization.")

    args = parser.parse_args()
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    args.expl2label_encoder_checkpt = torch.load(args.expl2label_model, map_location=args.device)['args'].encoder_checkpt
    
    # Set seed
    set_seed(args)

    # Load ExplainThenPredictModel
    model = ExplainThenPredictModel(args).to(args.device)

    tokenizer = AutoTokenizer.from_pretrained(model.seq2seq_model.config.encoder._name_or_path)

    # Load dataset
    test_data = EsnliDataset(args.test_data, rows=args.num_samples)

    test_dataloader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=SequentialSampler(test_data)
    )

    # Set up output files and directories
    if not args.output_dir.endswith('/'):
        args.output_dir += "/"
    
    if not args.generation_config_dir.endswith('/'):
        args.generation_config_dir += "/"

    os.makedirs(args.output_dir, exist_ok=True)

    # Setup logging
    args.logger = logging.getLogger(__name__)

    logging.basicConfig(filename=f"{args.output_dir}/output.log",
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    
    print(f"transformers: v. {transformers.__version__}\n")
    args.logger.info(f"transformers: v. {transformers.__version__}\n")

    print(f"Command line arguments:")
    args.logger.info(f"Command line arguments:")
    for name in vars(args):
        value = vars(args)[name]

        print(f"\t{name}: {value}")
        args.logger.info(f"\t{name}: {value}")
    
    # Get all generation config files in the given directory
    generation_config_files = [filename for filename in os.listdir(args.generation_config_dir) if os.path.isfile(os.path.join(args.generation_config_dir, filename))]

    for generation_config_file in generation_config_files:
        os.makedirs(args.output_dir + generation_config_file, exist_ok=True)

        scores_file = args.output_dir + generation_config_file + "/scores.out"
        predictions_file = args.output_dir + generation_config_file + "/predictions.csv"

        if os.path.exists(predictions_file):
            os.remove(predictions_file)
        
        if os.path.exists(scores_file):
            os.remove(scores_file)

        # Instantiate GenerationConfig class based on generation config file    
        with open(args.generation_config_dir + generation_config_file) as f:
            config_dict = json.load(f)
        
        if config_dict['early_stopping'] in ['True', 'False']:
            config_dict['early_stopping'] = eval(config_dict['early_stopping'])
        
        config_dict['do_sample'] = eval(config_dict['do_sample'])
        
        if config_dict['penalty_alpha'] == 'None':
            config_dict['penalty_alpha'] = None
        
        generation_config = GenerationConfig(**config_dict)

        # Evaluate ExplainThenPredict model on e-SNLI test data

        headers = [
            'premise', 
            'hypothesis', 
            'pred_explanation', 
            'pred_label', 
            'explanation_1', 
            'explanation_2', 
            'explanation_3', 
            'gold_label'
        ]

        csv_file = open(predictions_file, 'x')
        writer = csv.writer(csv_file)
        writer.writerow(headers)

        ID2LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
        LABEL2ID = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

        with torch.no_grad():
            for step, batch in enumerate(test_dataloader, 1):
                encoder_input = tokenizer(
                    batch['premise'], 
                    batch['hypothesis'],
                    padding='max_length', 
                    max_length=args.encoder_max_len, 
                    truncation=True, 
                    add_special_tokens=True,
                    return_tensors='pt'
                )
                
                input_ids = encoder_input['input_ids']
                attention_mask = encoder_input['attention_mask']

                pred_expls, pred_labels = model(input_ids, attention_mask, generation_config)

                pred_labels = pred_labels.argmax(dim=-1)
                gold_labels = batch['label'].to(args.device)
                            
                for i in range(len(pred_expls)):
                    row = []

                    row.append(batch['premise'][i])
                    row.append(batch['hypothesis'][i])
                    row.append(pred_expls[i])
                    row.append(ID2LABEL[pred_labels[i].item()])
                    row.append(batch['explanation_1'][i])
                    row.append(batch['explanation_2'][i])
                    row.append(batch['explanation_3'][i])
                    row.append(ID2LABEL[gold_labels[i].item()])

                    writer.writerow(row)

        csv_file.close()
        
        sentences_dict = preprocess(predictions_file, args)
        predictions = sentences_dict['predictions']
        references = sentences_dict['references']

        nlg_metrics_dict = compute_nlg_metrics(predictions, references)

        bleu_score = nlg_metrics_dict['bleu_score']
        meteor_score = nlg_metrics_dict['meteor_score']
        rouge_1_score, rouge_2_score = nlg_metrics_dict['rouge_1_score'], nlg_metrics_dict['rouge_2_score']
        bert_score = nlg_metrics_dict['bert_score']

        acc_dict = compute_acc(predictions_file, ID2LABEL, LABEL2ID)

        entailment_acc = acc_dict['entailment']
        contradiction_acc = acc_dict['contradiction']
        neutral_acc = acc_dict['neutral']
        acc = acc_dict['total']

        with open(scores_file, 'w') as f:
            f.write('==================== NLG metrics ======================\n')
            f.write(f'BLEU score: {bleu_score:.4f}\n')
            f.write(f'METEOR score: {meteor_score:.4f}\n')
            f.write(f'ROUGE-1 score: {rouge_1_score:.4f}\n')
            f.write(f'ROUGE-2 score: {rouge_1_score:.4f}\n')
            f.write(f'BERT score: {bert_score:.4f}\n\n')

            f.write('================ Classification metrics ===============\n')
            f.write(f'Total accuracy: {acc:.4f}\n')
            f.write(f'Entailment class accuracy: {entailment_acc:.4f}\n')
            f.write(f'Contradiction class accuracy: {contradiction_acc:.4f}\n')
            f.write(f'Neutral class accuracy: {neutral_acc:.4f}\n')