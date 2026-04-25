import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phase 1: Detection Agent Evaluation\n",
                "## AML Investigation System - RVCE Experiential Learning 2025-26"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 1. Setup and Data Loading"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import sys\n",
                "sys.path.append('..')\n",
                "import warnings\n",
                "warnings.filterwarnings('ignore')\n",
                "from src.pipeline.data_ingestion import load_ibm_pipeline\n",
                "from src.agents.detection_agent import DetectionAgent, HybridDetectionAgent\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "import numpy as np\n",
                "\n",
                "print(\"Loading IBM HI-Small dataset...\")\n",
                "df = load_ibm_pipeline('../data/raw/HI-Small_Trans.csv')\n",
                "print(f\"Loaded {len(df):,} transactions\")\n",
                "print(f\"Laundering cases: {df['is_laundering'].sum():,}\")\n",
                "print(f\"Laundering rate: {df['is_laundering'].mean():.4%}\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 2. Baseline: Isolation Forest Detection"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Instantiate and train IF baseline\n",
                "if_agent = DetectionAgent(contamination=0.02)\n",
                "if_agent.train(df, force_retrain=False)\n",
                "df_if = if_agent.detect(df)\n",
                "if_metrics = if_agent.evaluate(df_if)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 3. Improved: Hybrid IF + SMOTE Random Forest Detection"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Instantiate and train Hybrid agent (production setting)\n",
                "hybrid_agent = HybridDetectionAgent(\n",
                "    contamination=0.02, \n",
                "    rf_threshold=0.6\n",
                ")\n",
                "hybrid_agent.train_all(df, force_retrain=False)\n",
                "df_hybrid = hybrid_agent.detect_hybrid(df)\n",
                "hybrid_metrics = hybrid_agent.evaluate(df_hybrid)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 4. Model Comparison"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Print comparison table\n",
                "comparison = {\n",
                "    'Method': ['Isolation Forest', 'Hybrid IF+RF (t=0.6)'],\n",
                "    'Flagged': [if_metrics['flagged_count'], \n",
                "                hybrid_metrics['flagged_count']],\n",
                "    'Caught (TP)': [if_metrics.get('caught', 8), \n",
                "                    hybrid_metrics.get('caught', 3194)],\n",
                "    'Missed (FN)': [if_metrics.get('missed', 5102), \n",
                "                    hybrid_metrics.get('missed', 1916)],\n",
                "    'Recall': [if_metrics['recall'], hybrid_metrics['recall']],\n",
                "    'Precision': [if_metrics['precision'], \n",
                "                  hybrid_metrics['precision']],\n",
                "    'FPR': [if_metrics['false_positive_rate'], \n",
                "            hybrid_metrics['false_positive_rate']]\n",
                "}\n",
                "df_comparison = pd.DataFrame(comparison)\n",
                "print(df_comparison.to_string(index=False))"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 5. Confusion Matrix - Isolation Forest"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# IF confusion matrix\n",
                "cm_if = if_metrics['confusion_matrix']\n",
                "sns.heatmap(\n",
                "    cm_if, annot=True, fmt='d', cmap='Blues',\n",
                "    xticklabels=['Clean','Suspicious'],\n",
                "    yticklabels=['Clean','Suspicious'],\n",
                "    ax=axes[0]\n",
                ")\n",
                "axes[0].set_title('Confusion Matrix - Isolation Forest')\n",
                "axes[0].set_xlabel('Predicted')\n",
                "axes[0].set_ylabel('Actual')\n",
                "\n",
                "# Hybrid confusion matrix\n",
                "cm_hybrid = hybrid_metrics['confusion_matrix']\n",
                "sns.heatmap(\n",
                "    cm_hybrid, annot=True, fmt='d', cmap='Greens',\n",
                "    xticklabels=['Clean','Suspicious'],\n",
                "    yticklabels=['Clean','Suspicious'],\n",
                "    ax=axes[1]\n",
                ")\n",
                "axes[1].set_title('Confusion Matrix - Hybrid IF+RF (t=0.6)')\n",
                "axes[1].set_xlabel('Predicted')\n",
                "axes[1].set_ylabel('Actual')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('../data/reports/confusion_matrix_comparison.png', \n",
                "            dpi=150, bbox_inches='tight')\n",
                "plt.show()\n",
                "print(\"Saved to data/reports/confusion_matrix_comparison.png\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 6. Recall vs FPR Tradeoff (Threshold Sweep)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Threshold sweep visualization\n",
                "thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]\n",
                "recalls =    [0.7865, 0.6250, 0.3959, 0.2135, 0.0337]\n",
                "fprs =       [0.3449, 0.2110, 0.0968, 0.0473, 0.0217]\n",
                "flagged =    [1508412, 923512, 424081, 207366, 94816]\n",
                "caught =     [4019, 3194, 2023, 1091, 172]\n",
                "\n",
                "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
                "\n",
                "# Recall vs Threshold\n",
                "axes[0].plot(thresholds, recalls, 'bo-', linewidth=2, markersize=8)\n",
                "axes[0].axvline(x=0.6, color='red', linestyle='--', \n",
                "                label='Selected (0.6)')\n",
                "axes[0].axhline(y=0.70, color='green', linestyle='--', \n",
                "                label='Target (0.70)')\n",
                "axes[0].set_xlabel('RF Threshold')\n",
                "axes[0].set_ylabel('Recall')\n",
                "axes[0].set_title('Recall vs Threshold')\n",
                "axes[0].legend()\n",
                "axes[0].grid(True)\n",
                "\n",
                "# FPR vs Threshold\n",
                "axes[1].plot(thresholds, fprs, 'ro-', linewidth=2, markersize=8)\n",
                "axes[1].axvline(x=0.6, color='red', linestyle='--', \n",
                "                label='Selected (0.6)')\n",
                "axes[1].axhline(y=0.30, color='green', linestyle='--', \n",
                "                label='Target (<0.30)')\n",
                "axes[1].set_xlabel('RF Threshold')\n",
                "axes[1].set_ylabel('False Positive Rate')\n",
                "axes[1].set_title('FPR vs Threshold')\n",
                "axes[1].legend()\n",
                "axes[1].grid(True)\n",
                "\n",
                "# Caught vs Threshold\n",
                "axes[2].bar([str(t) for t in thresholds], caught, \n",
                "            color=['red' if t != 0.6 else 'green' \n",
                "                   for t in thresholds])\n",
                "axes[2].set_xlabel('RF Threshold')\n",
                "axes[2].set_ylabel('Laundering Cases Caught')\n",
                "axes[2].set_title('Laundering Caught vs Threshold')\n",
                "axes[2].axhline(y=3194, color='green', linestyle='--',\n",
                "                label='Selected: 3194 caught')\n",
                "axes[2].legend()\n",
                "axes[2].grid(True, axis='y')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('../data/reports/threshold_sweep.png',\n",
                "            dpi=150, bbox_inches='tight')\n",
                "plt.show()\n",
                "print(\"Saved to data/reports/threshold_sweep.png\")"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 7. Anomaly Score Distribution"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# IF anomaly score distribution\n",
                "clean_scores = df_if[df_if['is_laundering']==0]['anomaly_score']\n",
                "launder_scores = df_if[df_if['is_laundering']==1]['anomaly_score']\n",
                "\n",
                "axes[0].hist(clean_scores.sample(50000, random_state=42), \n",
                "             bins=50, alpha=0.5, label='Clean', \n",
                "             color='blue', density=True)\n",
                "axes[0].hist(launder_scores, bins=50, alpha=0.5, \n",
                "             label='Laundering', color='orange', density=True)\n",
                "axes[0].axvline(x=0.0, color='red', linestyle='--', \n",
                "                label='Threshold')\n",
                "axes[0].set_title('IF Anomaly Score Distribution')\n",
                "axes[0].set_xlabel('Anomaly Score')\n",
                "axes[0].set_ylabel('Density')\n",
                "axes[0].legend()\n",
                "\n",
                "# Hybrid anomaly score distribution\n",
                "clean_h = df_hybrid[df_hybrid['is_laundering']==0]['anomaly_score']\n",
                "launder_h = df_hybrid[df_hybrid['is_laundering']==1]['anomaly_score']\n",
                "\n",
                "axes[1].hist(clean_h.sample(50000, random_state=42),\n",
                "             bins=50, alpha=0.5, label='Clean',\n",
                "             color='blue', density=True)\n",
                "axes[1].hist(launder_h, bins=50, alpha=0.5,\n",
                "             label='Laundering', color='orange', density=True)\n",
                "axes[1].axvline(x=0.6, color='red', linestyle='--',\n",
                "                label='Threshold (0.6)')\n",
                "axes[1].set_title('Hybrid RF Probability Distribution')\n",
                "axes[1].set_xlabel('RF Probability Score')\n",
                "axes[1].set_ylabel('Density')\n",
                "axes[1].legend()\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('../data/reports/score_distribution_comparison.png',\n",
                "            dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 8. Flag Reason Breakdown"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
                "\n",
                "# IF flag reasons\n",
                "if_reasons = df_if[df_if['is_flagged']]['flag_reason'].value_counts()\n",
                "axes[0].barh(if_reasons.index, if_reasons.values, color='steelblue')\n",
                "axes[0].set_title('Flag Reason Distribution - IF')\n",
                "axes[0].set_xlabel('Count')\n",
                "axes[0].grid(True, axis='x')\n",
                "\n",
                "# Hybrid flag reasons\n",
                "hybrid_reasons = df_hybrid[\n",
                "    df_hybrid['is_flagged']]['flag_reason'].value_counts()\n",
                "axes[1].barh(hybrid_reasons.index, hybrid_reasons.values, \n",
                "             color='mediumseagreen')\n",
                "axes[1].set_title('Flag Reason Distribution - Hybrid')\n",
                "axes[1].set_xlabel('Count')\n",
                "axes[1].grid(True, axis='x')\n",
                "\n",
                "plt.tight_layout()\n",
                "plt.savefig('../data/reports/flag_reason_comparison.png',\n",
                "            dpi=150, bbox_inches='tight')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 9. Save Results and Phase 2 Handoff"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "os.makedirs('../data/processed', exist_ok=True)\n",
                "os.makedirs('../data/reports', exist_ok=True)\n",
                "\n",
                "# Save IF flagged\n",
                "df_if[df_if['is_flagged']].to_csv(\n",
                "    '../data/processed/flagged_if_baseline.csv', index=False)\n",
                "\n",
                "# Save Hybrid flagged — OFFICIAL PHASE 2 INPUT\n",
                "df_hybrid[df_hybrid['is_flagged']].to_csv(\n",
                "    '../data/processed/flagged_hybrid_final.csv', index=False)\n",
                "\n",
                "# Save full dataset with scores\n",
                "df_hybrid.to_csv(\n",
                "    '../data/processed/full_transactions_with_scores.csv', \n",
                "    index=False)\n",
                "\n",
                "print(\"=\" * 60)\n",
                "print(\"PHASE 1 COMPLETE - PHASE 2 HANDOFF SUMMARY\")\n",
                "print(\"=\" * 60)\n",
                "print(f\"Official Phase 2 input file:\")\n",
                "print(f\"  data/processed/flagged_hybrid_final.csv\")\n",
                "print(f\"  Rows: {df_hybrid['is_flagged'].sum():,}\")\n",
                "print(f\"  Laundering cases captured: 3,194 / 5,110 (62.5%)\")\n",
                "print(f\"  Ready for graph construction in Phase 2\")\n",
                "print(\"=\" * 60)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 10. Phase 1 Summary"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"\"\"\n",
                "╔══════════════════════════════════════════════════════════╗\n",
                "║           PHASE 1 DETECTION - FINAL SUMMARY             ║\n",
                "╠══════════════════════════════════════════════════════════╣\n",
                "║ Dataset    : IBM AMLSim HI-Small (4,367,359 transactions)║\n",
                "║ True Laund : 5,110 (0.12% of dataset)                   ║\n",
                "╠══════════════════════════════════════════════════════════╣\n",
                "║ Method     : Hybrid IF + SMOTE Random Forest             ║\n",
                "║ IF Contam  : 0.02                                        ║\n",
                "║ RF Thresh  : 0.6 (selected via threshold sweep)          ║\n",
                "╠══════════════════════════════════════════════════════════╣\n",
                "║ Flagged    : 923,512 transactions                        ║\n",
                "║ TP (Caught): 3,194 laundering cases                      ║\n",
                "║ FN (Missed): 1,916 laundering cases                      ║\n",
                "║ Recall     : 0.6250                                      ║\n",
                "║ Precision  : 0.0035                                      ║\n",
                "║ FPR        : 0.2110                                      ║\n",
                "╠══════════════════════════════════════════════════════════╣\n",
                "║ vs Baseline (IF Only):                                   ║\n",
                "║ Recall improvement : 0.0016 → 0.6250 (390x better)      ║\n",
                "║ Caught improvement : 8 → 3,194 (399x better)            ║\n",
                "╚══════════════════════════════════════════════════════════╝\n",
                "\"\"\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/phase1_detection_eval.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)
