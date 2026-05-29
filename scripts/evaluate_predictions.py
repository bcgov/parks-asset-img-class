# scripts/evaluate_predictions.py
"""
Compute evaluation metrics for VLM predictions and log to MLflow.

Usage:
    python scripts/evaluate_predictions.py \
        --predictions results/vlm_predictions_stairs_gemini.csv \
        --ground_truth data/processed/test/attr_number_of_steps_test.csv \
        --attribute number_of_steps \
        --model gemini-3-flash-preview \
        --asset_type Stairs \
        --prompt_version v1
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import mlflow
from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
from src.mlflow_utils import setup_mlflow, make_run_name, make_standard_tags

import dagshub
dagshub.init(repo_owner='sgauth01', repo_name='parks-asset-img-class', mlflow=True)

def evaluate(predictions_path, ground_truth_path, attribute, model_name, asset_type, prompt_version):
    
    preds_df = pd.read_csv(predictions_path)
    gt_df = pd.read_csv(ground_truth_path)
    
    # merge on asset_id
    merged = preds_df.merge(
        gt_df[["asset_id", attribute]],
        on="asset_id",
        how="inner"
    )
    
    # drop rows where prediction failed or ground truth missing
    attr_key = attribute.replace("attr_", "").replace(",", "")
    col = f"{attr_key}_value"

    merged = merged[merged[col].notna()]
    merged = merged[merged[attribute].notna()]

    merged = merged.drop_duplicates("asset_id")

    # get valid labels from ground truth
    valid_labels = merged[attribute].dropna().unique().tolist()

    # filter predictions to only valid labels
    merged = merged[merged[col].isin(valid_labels)]
    
    y_true = merged[attribute].tolist()
    y_pred = merged[col].tolist()
    
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    n_samples = len(y_true)
    
    print(f"\n=== {attribute} | {model_name} ===")
    print(f"Samples evaluated: {n_samples}")
    print(f"Macro-F1:    {macro_f1:.3f}")
    print(f"Weighted-F1: {weighted_f1:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # log to MLflow using MLFlow helpers
    setup_mlflow()
    
    mlflow.autolog()
    
    with mlflow.start_run(
        run_name=make_run_name(attribute, model_name),
        tags=make_standard_tags(
            task=attribute,
            model_family=model_name.split("-")[0],
            model_name=model_name,
            data_version="v1",
            split_seed=48,
            extra={
                "asset_type": asset_type,
                "prompt_version": prompt_version
            }
        )
    ):
        mlflow.log_metric("macro_f1", macro_f1)
        mlflow.log_metric("weighted_f1", weighted_f1)
        mlflow.log_metric("n_samples", n_samples)
        mlflow.log_metric("macro_precision", macro_precision)
        mlflow.log_metric("weighted_precision", weighted_precision)
        mlflow.log_metric("macro_recall", macro_recall)
        mlflow.log_metric("weighted_recall", weighted_recall)
        mlflow.log_param("attribute", attribute)
        mlflow.log_param("model", model_name)
        mlflow.log_param("asset_type", asset_type)
        mlflow.log_param("prompt_version", prompt_version)
        mlflow.log_artifact(predictions_path)
    
    return {
        "attribute": attribute,
        "model": model_name,
        "asset_type": asset_type,
        "prompt_version": prompt_version,
        "n_samples": n_samples,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground_truth_dir", required=True, 
                        help="Directory containing attribute-specific ground truth CSVs")
    parser.add_argument("--attributes", required=True, nargs="+",
                        help="One or more attributes to evaluate")
    parser.add_argument("--model", required=True)
    parser.add_argument("--asset_type", default="unknown")
    parser.add_argument("--prompt_version", default="v1")
    args = parser.parse_args()
    
    all_results = []

    for attribute in args.attributes:
        
        filename = (attribute
                    .replace("attr_", "")
                    .replace(",", "")
                    .replace(" ", "_")
                    .replace("(", "")
                    .replace(")", "")
                    .replace("<", "lt")
                    .replace(">", "gt")
                    .replace("/", "_")
        )

        # only add attr_ prefix for actual attr_ attributes
        if attribute.startswith("attr_"):
            ground_truth_path = f"{args.ground_truth_dir}/attr_{filename}_train.csv"
        else:
            ground_truth_path = f"{args.ground_truth_dir}/{filename}_train.csv"
        
        result = evaluate(
            predictions_path=args.predictions,
            ground_truth_path=ground_truth_path,
            attribute=attribute,
            model_name=args.model,
            asset_type=args.asset_type,
            prompt_version=args.prompt_version
        )

        if result is not None:
            all_results.append(result)

    # save all results to CSV
    if all_results:
        os.makedirs("results/vlm_eval_results", exist_ok=True)
        results_df = pd.DataFrame(all_results)
        output_filename = f"results/vlm_eval_results/{args.model}_{args.asset_type}_{args.prompt_version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(output_filename, index=False)
        print(f"\nResults saved to: {output_filename}")