import argparse
import os
import numpy as np
import sys
import pickle
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from loguru import logger
from typing import Generator, List, Tuple
import random

def init_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)

def _get_top_k_efficient(scores_row, k: int):
    if len(scores_row) <= k:
        top_k_idx = np.argsort(-scores_row)[:k]
    else:
        top_k_idx = np.argpartition(-scores_row, kth=k-1)[:k]
        top_k_idx = top_k_idx[np.argsort(-scores_row[top_k_idx])]
    
    top_k_scores = scores_row[top_k_idx]
    return top_k_idx, top_k_scores

def _yield_pkl_candidates(pkl_data: dict, n_candidates: int) -> Generator[Tuple[int, int, float, List[int], List[float]], None, None]:
    if 'test_probs' not in pkl_data or 'test_labels' not in pkl_data:
        raise ValueError("PKL file must contain 'test_probs' and 'test_labels'")
    
    probs = pkl_data['test_probs']
    labels = pkl_data['test_labels']
    
    num_users = len(probs)
    if len(labels) != num_users:
        logger.warning(f"Mismatch in length: probs ({len(probs)}) vs labels ({len(labels)})")
    
    logger.info(f"Processing {num_users} users from PKL...")
    
    for user_idx in range(num_users):
        user_id = user_idx + 1
        target_item_id = labels[user_idx]
        user_scores = probs[user_idx]
        
        if not isinstance(user_scores, np.ndarray):
            user_scores = np.array(user_scores, dtype='float32')
        
        user_scores[0] = -np.inf
        
        target_score = 0.0
        if target_item_id < len(user_scores):
            target_score = float(user_scores[target_item_id])

        top_k_indices, top_k_scores = _get_top_k_efficient(user_scores, n_candidates)
        
        yield user_id, target_item_id, target_score, top_k_indices.tolist(), top_k_scores.tolist()

def _write_results(candidate_generator: Generator[Tuple[int, int, float, List[int], List[float]], None, None], output_path: str, n_candidates: int):
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    
    with open(output_path, 'w') as f:
        header_items = [f"item_{i+1}" for i in range(n_candidates)]
        header = f"user_id,gt_item,{','.join(header_items)}\n"
        f.write(header)
        
        for user_id, target_item_id, target_score, candidate_ids, candidate_scores in candidate_generator:
            candidates_str = ",".join(map(str, candidate_ids))
            current_len = len(candidate_ids)
            if current_len < n_candidates:
                padding = ",".join([""] * (n_candidates - current_len))
                if padding:
                    candidates_str += "," + padding
            
            f.write(f"{user_id},{target_item_id},{candidates_str}\n")
            
            processed_count += 1
            
    logger.success(f"Generated {processed_count} test samples")

def main():
    parser = argparse.ArgumentParser(description='Generate test CSV files with candidates directly from .pkl files.')
    parser.add_argument('--pkl_file', type=str, required=True,
                      help='Path to .pkl file containing precomputed predictions')
    parser.add_argument('--n_candidates', type=int, default=20,
                      help='Number of candidates (default: 20)')
    parser.add_argument('--seed', type=int, default=2020,
                      help='Random seed for reproducibility (default: 2020)')
    
    args = parser.parse_args()
    
    init_all_seeds(args.seed)
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rec_dir = os.path.join(base_dir, 'recommendations')
    
    pkl_path = Path(args.pkl_file)
    try:
        dataset_name = pkl_path.parent.name
        model_name = pkl_path.parent.parent.name.upper()
        
        if model_name in ['SAVED', 'TRAIN'] or dataset_name in ['saved', 'train']:
            raise ValueError("Path structure too shallow")
            
        filename = f"{dataset_name}-for-{model_name}.csv"
        logger.info(f"Inferred output filename: {filename}")
    except Exception:
        filename = f"test_candidates_from_{pkl_path.stem}.csv"
        logger.warning(f"Using fallback filename: {filename}")

    output_file = os.path.join(rec_dir, filename)
    
    if not os.path.exists(args.pkl_file):
        raise FileNotFoundError(f"PKL file not found: {args.pkl_file}")
    
    logger.info(f"Loading precomputed data from: {args.pkl_file}")
    with open(args.pkl_file, 'rb') as f:
        pkl_data = pickle.load(f)
    
    generator = _yield_pkl_candidates(pkl_data, args.n_candidates)
    _write_results(generator, output_file, args.n_candidates)
    
    logger.success(f"Output saved to: {output_file}")

if __name__ == '__main__':
    main()
