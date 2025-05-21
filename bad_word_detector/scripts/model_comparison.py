#!/usr/bin/env python
# bad_word_detector/scripts/model_comparison.py
"""
Script to evaluate and compare different trained models on a test dataset.
This helps identify which model performs best for toxic comment detection.
"""

import sys
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

# Add the parent directory to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from bad_word_detector.models.bert_model import BadWordDetector
from bad_word_detector.models.preprocessing import TextPreprocessor
from bad_word_detector.utils.logger import setup_logger
from bad_word_detector.utils.config import Config

logger = setup_logger()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare model performance")
    
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        required=True,
        help="Paths to model directories to compare"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(Config.DATA_DIR) / "comparison"),
        help="Directory to save comparison results"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="FrancophonIA/multilingual-hatespeech-dataset",
        help="HuggingFace dataset to use for testing"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold"
    )
    
    parser.add_argument(
        "--test_size",
        type=int,
        default=1000,
        help="Number of test examples to use"
    )
    
    return parser.parse_args()

def evaluate_model(model, texts, labels, threshold=0.5, model_name="Model"):
    """Evaluate a model on test data."""
    logger.info(f"Evaluating {model_name} on {len(texts)} examples")
    
    # Make predictions
    y_pred = []
    y_scores = []
    
    for text in texts:
        result = model.predict(text, threshold=threshold)
        y_pred.append(1 if result["is_toxic"] else 0)
        y_scores.append(result["confidence"])
    
    # Calculate metrics
    accuracy = accuracy_score(labels, y_pred)
    precision = precision_score(labels, y_pred, zero_division=0)
    recall = recall_score(labels, y_pred, zero_division=0)
    f1 = f1_score(labels, y_pred, zero_division=0)
    
    # Get classification report
    report = classification_report(labels, y_pred, output_dict=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, y_pred)
    
    # Calculate ROC and AUC
    fpr, tpr, _ = roc_curve(labels, y_scores)
    roc_auc = auc(fpr, tpr)
    
    results = {
        "model_name": model_name,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": float(roc_auc)
        }
    }
    
    logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, AUC: {roc_auc:.4f}")
    
    return results, y_pred, y_scores

def plot_comparison(results, output_dir):
    """Create comparative visualizations for model performance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model names and metrics
    model_names = [r["model_name"] for r in results]
    accuracy = [r["accuracy"] for r in results]
    precision = [r["precision"] for r in results]
    recall = [r["recall"] for r in results]
    f1 = [r["f1_score"] for r in results]
    auc_scores = [r["roc"]["auc"] for r in results]
    
    # Create bar chart comparing metrics
    metrics = pd.DataFrame({
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "AUC": auc_scores
    }, index=model_names)
    
    ax = metrics.plot(kind="bar", figsize=(12, 6))
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png")
    
    # Plot ROC curves
    plt.figure(figsize=(10, 8))
    
    for r in results:
        plt.plot(
            r["roc"]["fpr"], 
            r["roc"]["tpr"], 
            label=f"{r['model_name']} (AUC = {r['roc']['auc']:.3f})"
        )
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / "roc_comparison.png")
    
    # Save results as JSON
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(results, f, indent=4)
    
    logger.info(f"Comparison visualizations saved to {output_dir}")

def main():
    """Main comparison function."""
    args = parse_args()
    
    logger.info("Starting model comparison")
    
    # Load test data
    preprocessor = TextPreprocessor()
    _, _, test_texts, test_labels = preprocessor.load_huggingface_dataset(
        dataset_name=args.dataset,
        clean=True,
        test_size=1.0,  # Use all data for testing
        sample_size=args.test_size
    )
    
    logger.info(f"Loaded {len(test_texts)} test examples")
    
    # Evaluate each model
    all_results = []
    
    for i, model_path in enumerate(args.models):
        model_name = f"Model {i+1}: {Path(model_path).name}"
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = BadWordDetector()
        model.load(model_path)
        
        # Evaluate
        result, _, _ = evaluate_model(
            model, test_texts, test_labels, 
            threshold=args.threshold, model_name=model_name
        )
        
        all_results.append(result)
    
    # Plot comparison
    plot_comparison(all_results, args.output_dir)
    
    logger.info("Model comparison completed")

if __name__ == "__main__":
    main()
