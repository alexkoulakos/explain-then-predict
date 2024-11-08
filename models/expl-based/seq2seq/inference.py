import torch
import argparse
import csv
import logging
import os
import sys

from torch.utils.data import DataLoader, SequentialSampler
from transformers import EncoderDecoderModel, GenerationConfig

sys.path.append("../")

from tokenizer import Seq2SeqTokenizer
from helpers import *
from dataset import EsnliDataset
from generation_config import GENERATION_CONFIG

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--trained_model", type=str, required=True, 
                        help="Path to the fine-tuned encoder-decoder model directory.")
    
    parser.add_argument("--text_generation_strategy", required=True, type=str, 
                        help="Strategy to use for text generation during model validation. Strategy must be exactly one of: greedy_search, beam_search, top-k_sampling, top-p_sampling")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files.")

    # Non-required params
    parser.add_argument("--encoder_max_len", type=int, default=128, 
                        help="Max number of tokens the encoder can process.")
    parser.add_argument("--decoder_max_len", type=int, default=64, 
                        help="Max number of tokens the decoder can process.")
    
    parser.add_argument("--batch_size", type=int, default=16, 
                        help="Batch size for testing.")
    parser.add_argument("--num_test_samples", type=int, default=-1, 
                        help="Number of samples to use for testing (-1 corresponds to the entire test set).")
    
    parser.add_argument("--remove_punctuation", action="store_true",
                        help="Whether to remove punctuation when calculating text generation metrics.")
    
    parser.add_argument('--seed', type=int, default=123,
                        help="Random seed for initialization.")

    args = parser.parse_args()

    assert args.text_generation_strategy in [
        'greedy_search',
        'beam_search',
        'top-k_sampling',
        'top-p_sampling'
    ], "Please provide a valid text generation strategy. Valid strategies are: greedy_search, beam_search, top-k_sampling, top-p_sampling"

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up output directory and files
    os.makedirs(args.output_dir, exist_ok=True)

    scores_file = os.path.join(args.output_dir, "scores.out")
    predictions_file = os.path.join(args.output_dir, "predictions.csv")
    logging_file = os.path.join(args.output_dir, "output.log")
    
    # Setup logging
    args.logger = logging.getLogger(__name__)

    logging.basicConfig(filename=logging_file,
                        format = '%(asctime)s - %(levelname)s - %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)

    print(f"Command line arguments:")
    args.logger.info(f"Command line arguments:")

    for name in vars(args):
        value = vars(args)[name]

        print(f"\t{name}: {value}")
        args.logger.info(f"\t{name}: {value}")
    
    args.n_gpus = torch.cuda.device_count()

    # Set seed
    set_seed(args)

    # Load pretrained encoder-decoder model
    model = EncoderDecoderModel.from_pretrained(args.trained_model)

    model.to(args.device)

    # Extract encoder and decoder checkpoints from model config
    args.encoder_checkpt = model.config.encoder._name_or_path
    args.decoder_checkpt = model.config.decoder._name_or_path

    # Load tokenizer for Seq2Seq task
    tokenizer = Seq2SeqTokenizer(
        args.encoder_checkpt,
        args.decoder_checkpt,
        args.encoder_max_len,
        args.decoder_max_len
    )

    # args.text_generation_strategy points to the corresponding entry of GENERATION_CONFIG
    config_dict = GENERATION_CONFIG[args.text_generation_strategy]

    config_dict['decoder_start_token_id'] = tokenizer.decoder_tokenizer.bos_token_id
    config_dict['eos_token_id'] = tokenizer.decoder_tokenizer.eos_token_id
    config_dict['pad_token_id'] = tokenizer.decoder_tokenizer.pad_token_id

    # Instantiate GenerationConfig class based on config_dict
    generation_config = GenerationConfig(**config_dict)

    # Load dataset
    test_data = EsnliDataset("test", rows=args.num_test_samples)

    test_dataloader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        pin_memory=True, 
        shuffle=False, 
        sampler=SequentialSampler(test_data)
    )

    # Evaluate model on e-SNLI test data
    model.eval()

    headers = ['premise', 'hypothesis', 'label', 'pred_explanation', 'explanation_1', 'explanation_2', 'explanation_3']

    csv_file = open(predictions_file, mode='x', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(headers)
    
    with torch.no_grad():
        for step, batch in enumerate(test_dataloader, 1):
            encoder_input = tokenizer((batch['premise'], batch['hypothesis']))
            input_ids = encoder_input['input_ids'].to(args.device)
            attention_mask = encoder_input['attention_mask'].to(args.device)
            
            # Generate explanation
            output_ids = model.generate(
                input_ids, 
                attention_mask=attention_mask,
                generation_config=generation_config
            )

            pred_expls = tokenizer.batch_decode(output_ids)

            gold_labels = batch['label'].to(args.device)

            for i in range(len(pred_expls)):
                # pred_expls[i] = pred_expls[i].replace('\n', '')

                row = []

                row.append(batch['premise'][i])
                row.append(batch['hypothesis'][i])
                row.append(ID2LABEL[batch['label'][i].item()])
                row.append(pred_expls[i])
                row.append(batch['explanation_1'][i])
                row.append(batch['explanation_2'][i])
                row.append(batch['explanation_3'][i])

                writer.writerow(row)
        
        csv_file.close()
        
        sentences_dict = preprocess(predictions_file, args)
        metrics_dict = compute_metrics(sentences_dict['predictions'], sentences_dict['references'])

        bleu_score = metrics_dict['bleu_score']
        meteor_score = metrics_dict['meteor_score']
        rouge_1_score, rouge_2_score = metrics_dict['rouge_1_score'], metrics_dict['rouge_2_score']
        bert_score = metrics_dict['bert_score']

        with open(scores_file, 'w') as f:
            f.write(f'BLEU score: {bleu_score:.4f}\n')
            f.write(f'METEOR score: {meteor_score:.4f}\n')
            f.write(f'ROUGE-1 score: {rouge_1_score:.4f}\n')
            f.write(f'ROUGE-2 score: {rouge_1_score:.4f}\n')
            f.write(f'BERT score: {bert_score:.4f}\n')