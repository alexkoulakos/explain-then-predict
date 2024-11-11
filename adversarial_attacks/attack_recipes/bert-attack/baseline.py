import textattack
import torch
import argparse
import ssl
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from typing import List

from textattack.attacker import Attacker
from textattack.attack_args import AttackArgs

from textattack.transformations.word_swaps import WordSwapMaskedLM
from textattack.attack_recipes import BERTAttackLi2020

from models.baseline import BertNLIModel
from baseline_tokenizer import BertNLITokenizer
from model_wrappers.baseline import BertNLIModelWrapper

from utils import *

def construct_attacks(args) -> List[textattack.Attack]:
    attacks = []

    for max_candidates in args.max_candidates:
        # Bind BERT-attack attack recipe to model wrapper
        attack = BERTAttackLi2020.build(model_wrapper)

        # Create WordSwapMaskedLM transformation according to `max_candidates` parameter.
        attack.transformation = WordSwapMaskedLM(method="bert-attack", max_candidates=max_candidates)
        
        attacks.append(add_input_column_modification(attack, args.column_to_ignore))
    
    return attacks

if __name__ == '__main__':
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context

    parser = argparse.ArgumentParser()

    # Required params
    parser.add_argument("--bert_nli_trained_model", type=str, required=True, 
                        help="Path to fine-tuned baseline classifier model file")
    
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to store output files.")

    # Non-required params
    parser.add_argument("--encoder_max_len", type=int, default=128, 
                        help="Max number of tokens the encoder can process.")
    
    parser.add_argument("--max_candidates", nargs="+", type=int, default=0.5,
                        help="List of K values to use in each run.")
    
    parser.add_argument("--target_sentence", type=str, default="premise",
                        help="Input sentence to be perturbed (premise or hypothesis)")
    parser.add_argument("--num_samples", type=int, default=-1, 
                        help="Number of samples to attack (-1 corresponds to the entire test set)")
    
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for initialization")

    args = parser.parse_args()

    assert args.target_sentence in ['premise', 'hypothesis'], "Parameter `target_sentence` can only have the values 'premise' or 'hypothesis'"

    if args.target_sentence == 'premise':
        args.column_to_ignore = 'hypothesis'
    else:
        args.column_to_ignore = 'premise'

    # Extract checkpoint info from BERT-NLI trained model file
    # All trained BERT-NLI models are located in directories of the form `./train_results/{transformers_version}/{encoder_checkpoint}/encoder_max_length_{encoder_max_length}__batch_size_{batch_size}__n_gpus_{n_gpus}/model.pt`.
    args.bert_nli_encoder_checkpoint =  args.bert_nli_trained_model.split('/')[-3]
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load pretrained model
    model = BertNLIModel(
        args.bert_nli_encoder_checkpoint,
        args.device
    )

    model.to(args.device)

    if args.device == torch.device('cpu'):
        model.load_state_dict(torch.load(args.bert_nli_trained_model, map_location=args.device)['model_state_dict'])
    else:
        model.load_state_dict(torch.load(args.bert_nli_trained_model)['model_state_dict'])
    
    # Load tokenizer for BERT-NLI task
    tokenizer = BertNLITokenizer(
        args.bert_nli_encoder_checkpoint, 
        args.encoder_max_len
    )

    # Load full model via its model wrapper
    model_wrapper = BertNLIModelWrapper(model, tokenizer)

    # Specify victim dataset
    dataset = load_dataset()

    # Create attacks, based on the different values of max_candidates (K)
    attacks = construct_attacks(args)

    for attack, max_candidates in zip(attacks, args.max_candidates):
        current_output_dir = args.output_dir + f"max_candidates_{max_candidates}/"
        os.makedirs(current_output_dir, exist_ok=True)

        # Redirect stdout to a log file
        sys.stdout = open(current_output_dir + 'output.log', 'w')

        # Attack all test samples with CSV logging and checkpoint saved every 1000 intervals
        attack_args = AttackArgs(
            num_examples=args.num_samples,
            log_to_csv=current_output_dir + "results.csv",
            checkpoint_interval=1000,
            checkpoint_dir=current_output_dir + "checkpoints",
            disable_stdout=False,
            enable_advance_metrics=False,
            random_seed=args.seed
        )

        attacker = Attacker(attack, dataset, attack_args)
        attacker.attack_dataset()

        df = pd.read_csv(current_output_dir + "results.csv")

        with open(current_output_dir + 'results.txt', 'w') as f:
            for label in ['entailment', 'contradiction', 'neutral']:
                percent, percent_perturbed = analyze_attack_results(df, label)

                f.write(f"Percentage of {label} labels that were successfully attacked: {percent:.2f}%\n")

                for k in percent_perturbed.keys():
                    f.write(f'\tPercentage of rows with "perturbed_output" = {k}: {percent_perturbed[k]:.2f}%\n')
                f.write("\n")