import csv


def save_classification_report(report: dict, model_name: str):
    path = "./results/classification_metrics.csv"
    with open(path, "a") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(
                [
                    "model",
                    "precision",
                    "recall",
                    "f1-score",
                    "weighted-precision",
                    "weighted-recall",
                    "weighted-f1-score",
                ]
            )
        writer.writerow(
            [
                model_name,
                report["macro avg"]["precision"],
                report["macro avg"]["recall"],
                report["macro avg"]["f1-score"],
                report["weighted avg"]["precision"],
                report["weighted avg"]["recall"],
                report["weighted avg"]["f1-score"],
            ]
        )


def save_matching_accuracy(accuracy: float, model_name: str):
    path = "./results/matching_accuracy.csv"
    with open(path, "a") as f:
        writer = csv.writer(f)
        if f.tell() == 0:
            writer.writerow(["model", "accuracy"])
        writer.writerow([model_name, accuracy])
