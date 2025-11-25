import pandas as pd
import ast
import sys

def print_section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def parse_candidates(df):
    results = []
    for _, row in df.iterrows():
        cands = ast.literal_eval(row['candidate_item_id'])
        gt = row['item_id']
        gt_in = gt in cands
        pos = cands.index(gt) if gt_in else None
        results.append({'cands': cands, 'gt': gt, 'gt_in': gt_in, 'pos': pos})
    return results

def main():
    # Load data
    test_file = sys.argv[1] if len(sys.argv) > 1 else 'data/ml-100k/test_SGCL.csv'
    try:
        df = pd.read_csv(test_file)
    except FileNotFoundError:
        print(f"Error: File {test_file} not found.")
        return

    # Basic info
    print_section(f"FILE: {test_file}")
    print(f"Total samples: {len(df)}")
    
    results = parse_candidates(df)

    # Candidate counts
    cand_counts = [len(r['cands']) for r in results]
    print(f"\nCandidate count distribution:")
    print(pd.Series(cand_counts).value_counts().sort_index())
    
    # GT inclusion stats
    gt_in_count = sum(r['gt_in'] for r in results)
    gt_rate = gt_in_count / len(results) * 100
    print_section("GROUND TRUTH STATISTICS")
    print(f"GT in candidates: {gt_in_count}/{len(results)} ({gt_rate:.1f}%)")
    
    chunk_size = 1000
    best_split = None
    max_gt_count = -1
    
    print_section("SPLIT ANALYSIS")
    
    num_full_chunks = len(results) // chunk_size
    
    for i in range(num_full_chunks):
        start_idx = i * chunk_size
        end_idx = (i + 1) * chunk_size
            
        chunk_results = results[start_idx:end_idx]
        gt_in_count = sum(r['gt_in'] for r in chunk_results)
        
        split_name = f"{start_idx+1}-{end_idx}"
        print(f"Split {split_name}: {gt_in_count} GT users")
        
        if gt_in_count > max_gt_count:
            max_gt_count = gt_in_count
            best_split = split_name
    
    leftover = len(results) % chunk_size
    if leftover > 0:
        start_idx = len(results) - leftover
        end_idx = len(results)
        
        chunk_results = results[start_idx:end_idx]
        gt_in_count = sum(r['gt_in'] for r in chunk_results)
        
        split_name = f"{start_idx+1}-{end_idx}"
        print(f"Split {split_name}: {gt_in_count} GT users (final {leftover} samples)")
        
        if gt_in_count > max_gt_count:
            max_gt_count = gt_in_count
            best_split = split_name

    if best_split:
        print_section("RESULT")
        print(f"Split with most GT users: {best_split} ({max_gt_count} users)")
    else:
        print("\nNo valid splits found (all < 1000 samples).")

if __name__ == "__main__":
    main()
