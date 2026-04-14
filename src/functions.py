import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def roc_curve_visualization(
    y_test,
    y_prob,
    save_path: str = None,
    show: bool = False
) -> None:

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))

    plt.plot(fpr, tpr, label=f"Model (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()

    if save_path:
        plt.savefig(save_path)

    if show:
        plt.show()