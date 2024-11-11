import textattack
import datasets
import pandas as pd

from textattack.constraints.pre_transformation import InputColumnModification

def load_dataset():
    dataset = datasets.load_dataset("snli", split="test").filter(lambda example: example['label'] in [0, 1, 2])

    data = []

    for index, d in enumerate(dataset):
        input = (d['premise'], d['hypothesis'])
        output = d['label']
        
        data.append((input, output))
    
    return textattack.datasets.Dataset(data, input_columns=("premise", "hypothesis"))

# Add InputColumnModification constraint, so that perturbations happen on either premise or hypothesis.
def add_input_column_modification(attack, column_to_ignore):
    assert column_to_ignore in ['premise', 'hypothesis']

    pre_transformation_constraints = []

    new_pretransformation_constraint = InputColumnModification(
        matching_column_labels=['premise', 'hypothesis'], 
        columns_to_ignore={column_to_ignore}
    )

    for i in range(len(attack.pre_transformation_constraints)):
        if not isinstance(attack.pre_transformation_constraints[i], InputColumnModification):
            pre_transformation_constraints.append(attack.pre_transformation_constraints[i])
    
    pre_transformation_constraints.append(new_pretransformation_constraint)

    attack.pre_transformation_constraints = pre_transformation_constraints

    return attack

def analyze_attack_results(df: pd.DataFrame, original_label: str):
    assert original_label in ['entailment', 'contradiction', 'neutral']

    ID2LABEL = {0: 'entailment', 1: 'neutral', 2: 'contradiction'}
    LABEL2ID = {'entailment': 0, 'neutral': 1, 'contradiction': 2}

    original_id = LABEL2ID[original_label]
    perturbed_ids = list(set([0, 1, 2]).difference(set([original_id])))

    # Filter the DataFrame
    filtered_df = df[(df['result_type'] == 'Successful') & (df['original_output'] == original_id)]

    # Total number of filtered rows
    total_filtered_rows = len(filtered_df)
    
    # Percentage compared to the original dataframe
    percent = (len(filtered_df) / len(df[df['result_type'] == 'Successful'])) * 100

    percent_perturbed = dict()

    for perturbed_id in perturbed_ids:
        # Count rows for perturbed id
        count_perturbed = len(filtered_df[filtered_df['perturbed_output'] == perturbed_id])

        # Calculate the percentages
        percent_perturbed[ID2LABEL[perturbed_id]] = (count_perturbed / total_filtered_rows) * 100

    return percent, percent_perturbed