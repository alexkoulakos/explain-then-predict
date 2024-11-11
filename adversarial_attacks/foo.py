import torch
import ssl

import transformers
import textattack

if __name__ == '__main__':
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        # Legacy Python that doesn't verify HTTPS certificates by default
        pass
    else:
        # Handle target environment that doesn't support HTTPS verification
        ssl._create_default_https_context = _create_unverified_https_context
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = transformers.AutoModel.from_pretrained("textattack/bert-base-uncased-snli")
    model.to(device)

    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-uncased")

    model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

    dataset = textattack.datasets.HuggingFaceDataset("snli", split="test")

    attack = textattack.attack_recipes.CLARE2020.build(model_wrapper)

    attack_args = textattack.AttackArgs(num_examples=10)

    attacker = textattack.attacker.Attacker(attack, dataset, attack_args)
    attacker.attack_dataset()