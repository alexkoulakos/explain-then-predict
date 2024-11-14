# Adversarial attacks against the fine-tuned BERT-based classifier

## Example usage

### TextFooler
In order to perform an adversarial attack using the **TextFooler** attack recipe against the fine-tuned BERT-based NLI classifier (stored in directory *path/to/fine-tuned/bert-nli/model.pt*), you can run the following command:

```bash
python adversarial_attacks/expl-free/textfooler.py \
    --trained_model_file path/to/fine-tuned/bert-nli/model.pt \
    --encoder_checkpt bert-base-uncased \
    --output_dir path/to/output/dir \
    --encoder_max_len 128 \
    --max_candidates 50 \
    --min_cos_sim 0.7 0.75 \
    --target_sentence premise \
    --num_samples -1 \
    --seed 123
```

:information_source: Parameter `min_cos_sim` can receive multiple values from the command line. An adversarial attack is performed for every combination of the parameters `max_candidates` and `min_cos_sim`. In the example above, 2 adversarial attacks take place:

* The first adversarial attack has parameters `max_candidates=50` and `min_cos_sim=0.7` and all its related output files are stored in the directory `path/to/output/dir/max_candidates_50__min_cos_sim_0.7`.
* The second adversarial attack has parameters `max_candidates=50` and `min_cos_sim=0.75` and all its related output files are stored in the directory `path/to/output/dir/max_candidates_50__min_cos_sim_0.75`.

### BERT-attack
In order to perform an adversarial attack using the **BERT-attack** attack recipe against the fine-tuned BERT-based NLI classifier (stored in directory *path/to/fine-tuned/bert-nli/model.pt*), you can run the following command:
```bash
python adversarial_attacks/expl-free/bert-attack.py \
    --trained_model_file path/to/fine-tuned/bert-nli/model.pt \
    --encoder_checkpt bert-base-uncased \
    --output_dir path/to/output/dir \
    --encoder_max_len 128 \
    --max_candidates 6 8 \
    --target_sentence premise \
    --num_samples -1 \
    --seed 123
```

:information_source: Parameter `max_candidates` can receive multiple values from the command line and an adversarial attack is performed for every value. In the example above, 2 adversarial attacks take place:

* The first adversarial attack has parameters `max_candidates=6` and all its related output files are stored in the directory `path/to/output/dir/max_candidates_6`.
* The second adversarial attack has parameters `max_candidates=8` and all its related output files are stored in the directory `path/to/output/dir/max_candidates_8`.

## General notes
:information_source: Parameter `target_sentence` allows you to specify the input component (premise or hypothesis) that gets perturbed.

:information_source: Please, ignore all the verbose tensorflow warnings.

:information_source: For optimal performance, you are strongly encouraged to run the attacks on a GPU device.