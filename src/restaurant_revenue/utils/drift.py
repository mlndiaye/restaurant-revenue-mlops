"""
Data Drift detection utility.
Compares a new production dataset or batch of queries against the reference training set
using Kolmogorov-Smirnov tests (for numerical features) and Chi-Square tests (for categorical features).
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency


def detect_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> dict:
    """
    Compares the reference (training) dataset against the current (production) dataset
    and returns a dictionary containing p-values and drift flags for each feature.
    """
    report = {
        "metrics": {
            "reference_count": len(reference_df),
            "current_count": len(current_df),
            "threshold": threshold,
        },
        "drift_detected": False,
        "features": {}
    }
    
    # Identify common features (excluding target 'revenue' and metadata 'Id')
    ref_cols = set(reference_df.columns)
    curr_cols = set(current_df.columns)
    common_cols = list(ref_cols.intersection(curr_cols) - {"revenue", "Id"})
    
    drifted_features_count = 0
    
    for col in common_cols:
        # Check datatype to determine the appropriate statistical test
        is_numeric = pd.api.types.is_numeric_dtype(reference_df[col])
        
        if is_numeric:
            # Kolmogorov-Smirnov test for numerical features
            # Null hypothesis: both samples are drawn from the same continuous distribution
            statistic, p_value = ks_2samp(
                reference_df[col].dropna(),
                current_df[col].dropna()
            )
            drifted = p_value < threshold
            test_name = "Kolmogorov-Smirnov"
        else:
            # Chi-Square test of independence for categorical features
            ref_counts = reference_df[col].value_counts()
            curr_counts = current_df[col].value_counts()
            
            # Align frequencies
            all_categories = sorted(list(set(ref_counts.index).union(set(curr_counts.index))))
            ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
            curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]
            
            contingency_table = np.array([ref_freq, curr_freq])
            
            # Add small epsilon to avoid zero frequency divisions in case category doesn't exist in one set
            if np.any(contingency_table == 0):
                contingency_table = contingency_table + 0.5
                
            try:
                statistic, p_value, _, _ = chi2_contingency(contingency_table)
                drifted = p_value < threshold
                test_name = "Chi-Square"
            except Exception:
                # Fallback if contingency table is degenerate
                statistic, p_value = 0.0, 1.0
                drifted = False
                test_name = "Chi-Square (failed)"
                
        report["features"][col] = {
            "type": "numerical" if is_numeric else "categorical",
            "test": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": bool(drifted)
        }
        
        if drifted:
            drifted_features_count += 1
            
    # We consider the dataset drifted if at least 10% of features show significant drift
    drift_ratio = drifted_features_count / len(common_cols) if common_cols else 0.0
    report["metrics"]["drifted_features_count"] = drifted_features_count
    report["metrics"]["total_features_count"] = len(common_cols)
    report["metrics"]["drift_ratio"] = drift_ratio
    
    if drift_ratio >= 0.1:  # 10% drift threshold
        report["drift_detected"] = True
        
    return report


def main():
    # Setup paths
    project_root = Path(__file__).resolve().parents[3]
    ref_path = project_root / "data" / "processed" / "train_processed.csv"
    
    # For demonstration, we compare train against the test set
    current_path = project_root / "data" / "processed" / "test_processed.csv"
    report_output_path = project_root / "data" / "processed" / "drift_report.json"
    
    if not ref_path.exists() or not current_path.exists():
        print("Error: Reference or current processed data does not exist.")
        return
        
    print(f"Loading reference data: {ref_path}")
    ref_df = pd.read_csv(ref_path)
    
    print(f"Loading current data: {current_path}")
    curr_df = pd.read_csv(current_path)
    
    print("Running drift detection...")
    report = detect_drift(ref_df, curr_df)
    
    # Save report
    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Drift report saved to {report_output_path}")
    print(f"Drift Detected: {report['drift_detected']}")
    print(f"Drifted features: {report['metrics']['drifted_features_count']}/{report['metrics']['total_features_count']}")


if __name__ == "__main__":
    main()
