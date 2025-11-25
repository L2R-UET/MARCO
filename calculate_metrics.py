import pandas as pd
import numpy as np
import math
from pathlib import Path

def calculate_ndcg_at_k(gt_item, recommended_items, k):
    """
    Calculate NDCG@k for a single user
    Following the implementation in marco.evaluation.rank_metric.NDCGAt
    
    Args:
        gt_item: ground truth item id
        recommended_items: list of recommended item ids
        k: cutoff position
    
    Returns:
        NDCG@k score
    """
    try:
        # Find position of gt_item (1-indexed)
        label_pos = recommended_items[:k].index(gt_item) + 1
    except ValueError:
        # gt_item not found in top-k
        return 0.0
    
    # NDCG = 1 / log2(position + 1)
    return 1.0 / math.log2(label_pos + 1.0)

def calculate_hr_at_k(gt_item, recommended_items, k):
    """
    Calculate Hit Rate@k for a single user
    Following the implementation in marco.evaluation.rank_metric.HitRatioAt
    
    Args:
        gt_item: ground truth item id
        recommended_items: list of recommended item ids
        k: cutoff position
    
    Returns:
        1 if hit, 0 otherwise
    """
    return 1 if gt_item in recommended_items[:k] else 0

def calculate_metrics_for_file(csv_path, last_n_users=2000):
    """
    Calculate metrics for a recommendation CSV file
    
    Args:
        csv_path: path to the CSV file
        last_n_users: number of last users to evaluate (default: 2000)
    
    Returns:
        Dictionary with metric results
    """
    print(f"\nProcessing: {csv_path.name}")
    print(f"{'='*60}")
    
    # Read CSV file
    df = pd.read_csv(csv_path)
    
    df = df.tail(last_n_users)
    total_users = len(df)
    
    print(f"Evaluating last {total_users} users")
    
    ndcg_5_scores = []
    ndcg_10_scores = []
    ndcg_20_scores = []
    hr_5_scores = []
    hr_10_scores = []
    hr_20_scores = []
    
    # Calculate metrics for each user
    for idx, row in df.iterrows():
        gt_item = row['gt_item']
        
        recommended_items = []
        for i in range(1, 21):
            col_name = f'item_{i}'
            if col_name in row:
                recommended_items.append(row[col_name])
        
        # Calculate NDCG@k
        ndcg_5_scores.append(calculate_ndcg_at_k(gt_item, recommended_items, 5))
        ndcg_10_scores.append(calculate_ndcg_at_k(gt_item, recommended_items, 10))
        ndcg_20_scores.append(calculate_ndcg_at_k(gt_item, recommended_items, 20))
        
        # Calculate HR@k
        hr_5_scores.append(calculate_hr_at_k(gt_item, recommended_items, 5))
        hr_10_scores.append(calculate_hr_at_k(gt_item, recommended_items, 10))
        hr_20_scores.append(calculate_hr_at_k(gt_item, recommended_items, 20))
    
    # Calculate average metrics
    results = {
        'file': csv_path.name,
        'num_users': total_users,
        'NDCG@5': np.mean(ndcg_5_scores),
        'NDCG@10': np.mean(ndcg_10_scores),
        'NDCG@20': np.mean(ndcg_20_scores),
        'HR@5': np.mean(hr_5_scores),
        'HR@10': np.mean(hr_10_scores),
        'HR@20': np.mean(hr_20_scores)
    }
    
    return results

def main():
    recommendations_dir = Path('train/recommendations')
    
    if not recommendations_dir.exists():
        print(f"Error: Directory '{recommendations_dir}' not found!")
        return
    
    # Find all CSV files
    csv_files = list(recommendations_dir.glob('*.csv'))
    
    # Filter out ml-100k files
    csv_files = [f for f in csv_files if 'ml-100k' not in f.name.lower()]
    
    if not csv_files:
        print(f"No CSV files found in '{recommendations_dir}'")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)")
    print(f"Calculating metrics for last 2000 users in each file...")
    
    all_results = []
    
    for csv_path in sorted(csv_files):
        results = calculate_metrics_for_file(csv_path, last_n_users=2000)
        all_results.append(results)
    
    # Print summary table
    print(f"\n{'='*80}")
    print("RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Model/Dataset':<35} {'HR@5':<12} {'NDCG@5':<12} {'HR@10':<12} {'NDCG@10':<12}")
    print(f"{'-'*80}")
    for res in all_results:
        print(f"{res['file']:<35} {res['HR@5']:<12.6f} {res['NDCG@5']:<12.6f} {res['HR@10']:<12.6f} {res['NDCG@10']:<12.6f}")

if __name__ == "__main__":
    main()
