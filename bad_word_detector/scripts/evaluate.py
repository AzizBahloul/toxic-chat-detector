# bad_word_detector/scripts/evaluate.py
import sys
import argparse
from pathlib import Path
import time
import json
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

# Add the parent directory to sys.path to import from our package
sys.path.append(str(Path(__file__).parent.parent))

from models.bert_model import BadWordDetector
from models.preprocessing import TextPreprocessor
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate bad word detection model")

    parser.add_argument(
        "--data_path", type=str, help="Path to evaluation data file (CSV/TSV)"
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        default=Config.MODEL_PATH,
        help="Directory containing the trained model",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(Config.DATA_DIR) / "output"),
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=Config.DEFAULT_THRESHOLD,
        help="Classification threshold",
    )

    return parser.parse_args()


def evaluate_model(model, test_texts, test_labels, threshold=0.5):
    """
    Evaluate model on test data.

    Args:
        model: BadWordDetector model
        test_texts: List of test texts
        test_labels: List of test labels
        threshold: Classification threshold

    Returns:
        Dictionary with evaluation metrics
    """
    start_time = time.time()
    predictions = []
    raw_predictions = []

    logger.info(f"Evaluating model on {len(test_texts)} examples")

    # Make predictions
    for text in test_texts:
        result = model.predict(text, threshold=threshold)
        predictions.append(1 if result["is_toxic"] else 0)
        raw_predictions.append(result["confidence"])

    # Calculate metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    cm = confusion_matrix(test_labels, predictions)

    # Get per-class report
    class_report = classification_report(test_labels, predictions, output_dict=True)

    elapsed_time = time.time() - start_time

    # Prepare results
    results = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "inference_time_total": elapsed_time,
        "inference_time_per_example": elapsed_time / len(test_texts),
    }

    logger.info(f"Evaluation completed in {elapsed_time:.2f} seconds")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")

    return results, raw_predictions


def visualize_results(results, raw_predictions, test_labels, output_dir):
    """
    Create visualizations for evaluation results.

    Args:
        results: Evaluation results
        raw_predictions: Raw prediction scores
        test_labels: True labels
        output_dir: Directory to save visualizations
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Confusion matrix heatmap
    plt.figure(figsize=(8, 6))
    cm = results["confusion_matrix"]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Toxic", "Toxic"],
        yticklabels=["Non-Toxic", "Toxic"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png")

    # Distribution of prediction scores
    plt.figure(figsize=(8, 6))

    # Convert to numpy arrays for easier manipulation
    import numpy as np

    raw_predictions = np.array(raw_predictions)
    test_labels = np.array(test_labels)

    # Plot distributions for each class
    sns.histplot(
        raw_predictions[test_labels == 0],
        color="green",
        alpha=0.5,
        bins=20,
        label="Non-Toxic",
        kde=True,
    )
    sns.histplot(
        raw_predictions[test_labels == 1],
        color="red",
        alpha=0.5,
        bins=20,
        label="Toxic",
        kde=True,
    )

    plt.axvline(
        results.get("threshold", 0.5),
        color="black",
        linestyle="--",
        label=f"Threshold ({results.get('threshold', 0.5)})",
    )

    plt.title("Distribution of Prediction Scores by Class")
    plt.xlabel("Prediction Score")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png")

    # Save the metrics as a JSON file
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=4)

    logger.info(f"Visualizations saved to {output_dir}")


def main():
    """Main evaluation function."""
    args = parse_args()

    logger.info("Starting model evaluation")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize model
    logger.info(f"Loading model from {args.model_dir}")
    model = BadWordDetector()
    model.load(args.model_dir)

    # Load test data
    if args.data_path:
        logger.info(f"Loading test data from {args.data_path}")

        preprocessor = TextPreprocessor()
        _, _, test_texts, test_labels = preprocessor.load_dataset(
            file_path=args.data_path,
            test_size=1.0,  # Use all data for testing
        )
    else:
        # If no specific test data is provided, generate sample data
        from scripts.train import generate_sample_dataset

        sample_path = output_dir / "sample_test_data.csv"
        generate_sample_dataset(sample_path, n_samples=200)

        preprocessor = TextPreprocessor()
        _, _, test_texts, test_labels = preprocessor.load_dataset(
            file_path=str(sample_path), test_size=1.0
        )

    logger.info(f"Loaded {len(test_texts)} test examples")

    # Evaluate model
    results, raw_predictions = evaluate_model(
        model, test_texts, test_labels, threshold=args.threshold
    )

    # Add threshold to results
    results["threshold"] = args.threshold

    # Create visualizations
    visualize_results(results, raw_predictions, test_labels, output_dir)

    logger.info(f"Evaluation results saved to {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
