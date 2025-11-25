import pandas as pd
import argparse
import os

def calculate_hitrate(csv_file):
    print(f"Calculating HitRate for: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"File not found: {csv_file}")
        return

    df = pd.read_csv(csv_file)
    
    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    hits_at_20 = 0
    total_users = len(df)
    
    print(f"Total users: {total_users}")
    
    for index, row in df.iterrows():
        gt_item = row['gt_item']
        
        candidate_cols = [c for c in df.columns if c.startswith('item_')]
        candidates = row[candidate_cols].values
        
        if gt_item in candidates[:1]:
            hits_at_1 += 1
        if gt_item in candidates[:5]:
            hits_at_5 += 1
        if gt_item in candidates[:10]:
            hits_at_10 += 1
        if gt_item in candidates[:20]:
            hits_at_20 += 1
            
    hr_1 = hits_at_1 / total_users
    hr_5 = hits_at_5 / total_users
    hr_10 = hits_at_10 / total_users
    hr_20 = hits_at_20 / total_users
    
    print("-" * 30)
    print(f"HitRate@1:  {hr_1:.4f}")
    print(f"HitRate@5:  {hr_5:.4f}")
    print(f"HitRate@10: {hr_10:.4f}")
    print(f"HitRate@20: {hr_20:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_file', type=str, default=r"train\recommendations\yelp2020-for-LRU.csv")
    args = parser.parse_args()
    
    calculate_hitrate(args.csv_file)
