import pandas as pd
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import ast
import random

def load_all_csv(data_dir: Path) -> pd.DataFrame:
    all_csv_path = data_dir / "all.csv"
    if not all_csv_path.exists():
        raise FileNotFoundError(f"all.csv not found in {data_dir}")
    
    print(f"Loading all.csv from {all_csv_path}...")
    all_df = pd.read_csv(all_csv_path)
    return all_df

def load_recommendations_csv(rec_file: Path) -> pd.DataFrame:
    if not rec_file.exists():
        raise FileNotFoundError(f"Recommendations file not found: {rec_file}")
    
    print(f"Loading recommendations from {rec_file}...")
    rec_df = pd.read_csv(rec_file)
    return rec_df

def build_user_history_map(all_df: pd.DataFrame) -> Dict[int, Tuple[List, List]]:
    print("Building user history map...")
    user_history = {}
    
    # Group by user_id and collect items and ratings
    for user_id, group in all_df.groupby('user_id'):
        # Sort by timestamp to maintain chronological order
        group = group.sort_values('timestamp')
        items = group['item_id'].tolist()
        ratings = group['rating'].tolist()
        user_history[user_id] = (items, ratings)
    
    return user_history

def safe_parse_list(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        if val == '[]' or val == '':
            return []
        try:
            return ast.literal_eval(val)
        except:
            return []
    return []

def convert_recommendations(rec_df: pd.DataFrame, user_history: Dict) -> pd.DataFrame:
    print("Converting recommendations to complete format...")
    
    random.seed(2020)
    
    rows = []
    
    for idx, row in rec_df.iterrows():
        user_id = int(row['user_id'])
        gt_item = int(row['gt_item'])
        
        candidate_cols = [col for col in rec_df.columns 
                         if col not in ['user_id', 'gt_item']]
        candidates = []
        for col in candidate_cols:
            try:
                item = int(row[col])
                candidates.append(item)
            except (ValueError, TypeError):
                continue
        
        # Randomly shuffle candidate item list
        random.shuffle(candidates)
        
        if user_id in user_history:
            history_items, history_ratings = user_history[user_id]
        else:
            history_items, history_ratings = [], []
        
        # Leave rating empty
        rating = ''
        
        new_row = {
            'user_id': user_id,
            'item_id': gt_item,
            'rating': rating,
            'history_item_id': str(history_items),
            'history_rating': str(history_ratings),
            'candidate_item_id': str(candidates)
        }
        
        rows.append(new_row)
        
        if (idx + 1) % 5000 == 0:
            print(f"  Processed {idx + 1} rows...")
    
    return pd.DataFrame(rows)

def main():
    parser = argparse.ArgumentParser(
        description="Convert recommendations CSV to complete test.csv format"
    )
    parser.add_argument(
        "rec_file",
        type=Path,
        help="Path to recommendations CSV file (e.g., train/recommendations/yelp-2020-for-LightGCN.csv)"
    )
    parser.add_argument(
        "-d", "--dataset-dir",
        type=Path,
        default=None,
        help="Path to dataset directory containing all.csv (e.g., data/Yelp2020). "
             "If not provided, will try to infer from recommendations file name."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output CSV file path. If not provided, will be saved in same directory as input with '_converted' suffix."
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not args.rec_file.exists():
        print(f"ERROR: Recommendations file not found: {args.rec_file}", file=sys.stderr)
        sys.exit(1)
    
    # Determine dataset directory
    if args.dataset_dir:
        dataset_dir = args.dataset_dir
    else:
        # Try to infer from file name
        # e.g., "yelp-2020-for-LightGCN.csv" -> "Yelp2020"
        filename = args.rec_file.stem.lower()
        
        # Map common dataset names
        dataset_map = {
            'yelp-2020': 'Yelp2020',
            'yelp2020': 'Yelp2020',
            'yelp': 'Yelp2020',
            'beauty': 'Beauty',
            'amazon-beauty': 'Beauty',
            'electronics': 'Electronics',
            'amazon-electronics': 'Electronics',
            'video_games': 'Video_Games',
            'video-games': 'Video_Games',
            'ml-100k': 'ml-100k',
            'ml-1m': 'ml-1m',
            'yelp2018': 'Yelp2018',
        }
        
        dataset_name = None
        for key, value in dataset_map.items():
            if key in filename:
                dataset_name = value
                break
        
        if not dataset_name:
            print(f"ERROR: Could not infer dataset name from file: {args.rec_file.name}", 
                  file=sys.stderr)
            print("Please provide dataset directory with -d/--dataset-dir", file=sys.stderr)
            sys.exit(1)
        
        dataset_dir = args.rec_file.parent.parent.parent.parent / "data" / dataset_name
    
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Dataset directory: {dataset_dir}")
    
    # Load data
    try:
        all_df = load_all_csv(dataset_dir)
        rec_df = load_recommendations_csv(args.rec_file)
    except Exception as e:
        print(f"ERROR: Failed to load data: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        user_history = build_user_history_map(all_df)
    except Exception as e:
        print(f"ERROR: Failed to build user history: {e}", file=sys.stderr)
        sys.exit(1)
    
    try:
        result_df = convert_recommendations(rec_df, user_history)
    except Exception as e:
        print(f"ERROR: Failed to convert recommendations: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        filename = args.rec_file.stem
        model_name = None
        if "-for-" in filename:
            model_name = filename.split("-for-")[-1]
        elif "_for_" in filename:
            model_name = filename.split("_for_")[-1]
        
        if model_name:
            output_path = dataset_dir / f"test_{model_name}.csv"
        else:
            output_path = dataset_dir / f"test_{args.rec_file.stem}.csv"
    
    # Save result
    try:
        print(f"Saving converted CSV to {output_path}...")
        result_df.to_csv(output_path, index=False)
        print(f"Successfully converted {len(result_df)} rows")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"ERROR: Failed to save output: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
