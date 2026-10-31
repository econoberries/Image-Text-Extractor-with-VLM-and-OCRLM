import os
import re
import ast
import json
import argparse
import numpy as np
from tqdm import tqdm
from itertools import product
import pandas as pd

def generate_combinations(input_dict):
    """
    Function to generate all possible combinations of values from a dictionary.
    """
    kie_answer = input_dict
    if not isinstance(kie_answer, dict):
        kie_answer = kie_answer.strip('"')
        try:
            kie_answer = json.loads(kie_answer)
        except json.JSONDecodeError:
            try:
                kie_answer = ast.literal_eval(kie_answer)
                if not isinstance(kie_answer, dict):
                    kie_answer = ast.literal_eval(kie_answer)
            except (ValueError, SyntaxError):
                print(f"Unable to parse 'answers' field: {kie_answer}")
                return {}
        
        # Ensure the parsed result is a dictionary.
        if not isinstance(kie_answer, dict):
            print("Parsed 'answers' is still not a dictionary.")
            raise ValueError("Input could not be parsed into a dictionary.")
    
        keys = list(kie_answer.keys())
        
        value_lists = []
        for single_key in keys:
            sinlge_value = kie_answer[single_key]
            if not isinstance(sinlge_value, list):
                sinlge_value = [sinlge_value]
            value_lists.append(sinlge_value)
    
        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))
    
        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

        return result
    
    else:
        keys = list(input_dict.keys())
        value_lists = [input_dict[key] for key in keys]

        # Compute the Cartesian product of the value lists.
        combinations = list(product(*value_lists))

        # Create a dictionary for each combination of values.
        result = [dict(zip(keys, values)) for values in combinations]

    return result

def convert_str_to_dict(predict_str: str):
    """
    Parses the 'predict' string and returns a dictionary.
    Missing or unparseable content is handled gracefully.

    Parameters:
    - predict_str (str): The prediction string containing the output dict.

    Returns:
    - dict: A dictionary extracted from the predict string.
    """
    # Remove code fences like ```python\n...\n```
    code_fence_pattern = r'```(?:python|json)?\n(.*?)\n```'
    match = re.search(code_fence_pattern, predict_str, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
    else:
        content = predict_str.strip()

    data = {}
    success = False

    # try parsing with JSON
    try:
        data = json.loads(content)
        success = True
    except json.JSONDecodeError:
        pass

    # try parsing with ast.literal_eval
    if not success:
        try:
            data = ast.literal_eval(content)
            if isinstance(data, dict):
                success = True
        except (ValueError, SyntaxError):
            pass

    # try parsing with regex
    if not success:
        key_value_pattern = r'["\']?([\w\s]+)["\']?\s*[:=]\s*["\']?([^\n,"\'{}]+)["\']?'
        matches = re.findall(key_value_pattern, content)
        try:
            for key, value in matches:
                data[key.strip()] = value.strip()
        except:
            return {}

    if not data:
        return {}

    try:
        result = {k.strip(): str(v).strip() for k, v in data.items()}
    except:
        return {}
    return result

def process_kie_data(data_item):
    assert len(data_item["answers"]) == 1
    answers = data_item["answers"][0]
    while isinstance(answers, str):
        answers = ast.literal_eval(answers)
    if not isinstance(answers,dict):
        print(f"Unexpected answers type {type(answers)} for id {data_item.get('id')}: {answers!r}")
    answers = {k: v if isinstance(v, list) else [v] for k, v in answers.items()}
    answers = generate_combinations(answers)
    if type(answers)==list and len(answers) == 1:
        if not isinstance(data_item["predict"], str):
            data_item["score"] = 0
            data_item["tp"] = 0
            data_item["fp"] = 0
            data_item["fn"] = len(answers[0])
            data_item["missing"] = 0
            data_item["wrong"] = 0
        else:
            pred_kie_dict = convert_str_to_dict(data_item["predict"])
            data_item["score"], data_item["fn"], data_item["fp"], data_item["tp"], data_item["missing"], data_item["wrong"] = compute_f1_score(pred_kie_dict, answers[0])
    else:
        max_score = -1
        for answer in answers:
            pred_kie_dict = convert_str_to_dict(data_item["predict"])
            score, fn, fp, tp, missing, wrong = compute_f1_score(pred_kie_dict, answer)               
            
            if score > max_score:
                max_score = score
                best_fn, best_fp, best_tp, best_missing, best_wrong = fn, fp, tp, missing, wrong  
        data_item["score"] = max_score
        data_item["fn"], data_item["fp"], data_item["tp"], data_item["missing"], data_item["wrong"] = best_fn, best_fp, best_tp, best_missing, best_wrong

def compute_f1_score(preds, gts, ignores=[]):
    """Compute the F1-score for KIE task between predicted and ground truth dictionaries.

    Args:
        preds (dict): The predicted key-value pairs.
        gts (dict): The ground truth key-value pairs.
        ignores (list): The list of keys to ignore during evaluation.

    Returns:
        dict: A dictionary where keys are field names and values are their corresponding F1-scores.
    """
    # Optionally remove ignored keys from predictions and ground truths
    keys = set(preds.keys()).union(set(gts.keys())) - set(ignores)
    f1_scores = {}

    fn = 0
    fp = 0
    tp = 0
    missing = 0
    wrong = 0

    for key in keys:
        pred_value = preds.get(key, None)
        gt_value = gts.get(key, None)

        if pred_value:
            pred_value = pred_value.lower().strip().replace("\n"," ").replace(" ", "")
        if gt_value:
            gt_value = gt_value.lower().strip().replace("\n"," ").replace(" ", "")

        if pred_value is None and gt_value is None:
            continue
        elif pred_value is None:
            # False Negative
            precision = 0.0
            recall = 0.0
            fn += 1
            missing += 1

        elif gt_value is None:
            # false positive
            precision = 0.0
            recall = 0.0
            fp += 1
        else:
            if pred_value == gt_value:
                # True positive
                precision = 1.0
                recall = 1.0
                tp +=1
            else:
                # False Negative
                precision = 0.0
                recall = 0.0
                fn += 1
                wrong +=1

        # Compute F1-score
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_scores[key] = f1_score

    if len(f1_scores) == 0:
        return 0, fn, fp, tp, missing, wrong  
    average_f1 = sum(f1_scores.values()) / len(f1_scores)

    return average_f1, fn, fp, tp, missing, wrong  

def main():

    df = pd.read_csv("data/aggregated_predictions_scored.csv")

    folders = ["e0fd2410-b1e1-4471-935f-29c3ff2dc58d", 
               "ac624162-b2cb-4f89-b087-4524b95980ef",
               "99d6fbe5-3cc5-4bf1-89dd-a607e5eab5d9",
               "8a14246e-d9e2-4e4e-ad6f-38fd0f25f44e",
               "3b821b76-66cc-427d-956e-20ccb9483f28",
               "34e930ce-85f0-41fb-9754-e457e53ae5f0"]

    records = []

    for folder in folders: 
        input_path = f"data/{folder}/predictions.json"
        with open(input_path, "r") as f:
            predict_file = json.load(f)

        for index, data_item in enumerate(tqdm(predict_file, desc=folder)):
            # adds score, tp, fp, fn to data_item
            process_kie_data(data_item)
            records.append({
                "run_id": folder,
                "id": data_item["id"],
                "recalc_score": data_item["score"],
                "true_pos": data_item["tp"],
                "false_pos": data_item["fp"],
                "false_neg": data_item["fn"],
                "missing": data_item["missing"],
                "wrong": data_item["wrong"]
            })

    results = pd.DataFrame(records)

    # Drop any stale versions of the columns, then merge on (run_id, id).
    new_cols = ["recalc_score", "true_pos", "false_pos", "false_neg", "missing", "wrong"]
    df = df.drop(columns=[c for c in new_cols if c in df.columns])
    df["id"] = df["id"].astype(str)
    results["id"] = results["id"].astype(str)
    df = df.merge(results, how="left", on=["run_id", "id"])

    df.to_csv("data/aggregated_predictions_rescored.csv", index=False)

    print("huzza")



if __name__ == "__main__":
    main()
