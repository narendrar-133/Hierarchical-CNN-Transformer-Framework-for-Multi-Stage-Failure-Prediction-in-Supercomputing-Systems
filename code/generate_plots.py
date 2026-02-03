import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
with open("complete_evaluation_results.pkl", "rb") as f:
    results = pickle.load(f)

preds = np.array(results["predictions"])
targets = np.array(results["targets"])
cm = np.array(results["confusion_matrix"])
label_names = ["Normal", "Pre-Failure", "Failure"]

# ===============================
# FIG 3 — Label Distribution
# ===============================
unique, counts = np.unique(targets, return_counts=True)

plt.figure()
plt.bar(label_names, counts)
plt.title("Label Distribution")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.savefig("fig3_label_distribution.png", dpi=300)
plt.close()

# ===============================
# FIG 4 — Confusion Matrix
# ===============================
plt.figure()
sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=300)
plt.close()

# ===============================
# FIG 5 — Prediction Tolerance Curve
# ===============================
tolerances = [0.25, 0.5, 0.75, 1.0]
acc = [np.mean(np.abs(preds - targets) <= t) * 100 for t in tolerances]

plt.figure()
plt.plot(tolerances, acc, marker='o')
plt.title("Prediction Accuracy vs Error Tolerance")
plt.xlabel("Tolerance (±)")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig("fig5_tolerance_curve.png", dpi=300)
plt.close()

# ===============================
# FIG 6 — True vs Predicted Scatter
# ===============================
plt.figure()
plt.scatter(targets, preds, s=1, alpha=0.3)
plt.title("True vs Predicted Severity")
plt.xlabel("True Severity")
plt.ylabel("Predicted Severity")
plt.tight_layout()
plt.savefig("fig6_true_vs_pred.png", dpi=300)
plt.close()

# ===============================
# FIG 7 — Risk Score Distribution
# ===============================
plt.figure()
for i, name in enumerate(label_names):
    plt.hist(preds[targets == i], bins=50, alpha=0.5, label=name)

plt.title("Predicted Risk Score Distribution")
plt.xlabel("Predicted Severity Score")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("fig7_risk_distribution.png", dpi=300)
plt.close()

print("All plots generated successfully.")
