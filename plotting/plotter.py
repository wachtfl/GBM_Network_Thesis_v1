import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

class Plotter:

    @staticmethod
    def plot_prediction_distribution(preds, patient_id, path):
        plt.figure(figsize=(8,4))
        sns.histplot(preds, bins=50)
        plt.title(f"Prediction Distribution – {patient_id}")
        plt.savefig(path)
        plt.close()

    @staticmethod
    def plot_auc_curve(labels, preds, path):
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = auc(fpr, tpr)

        plt.figure(figsize=(6,6))
        plt.plot(fpr, tpr, label=f"AUC = {auc_val:.3f}")
        plt.plot([0,1],[0,1], "k--")
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC")
        plt.savefig(path)
        plt.close()
