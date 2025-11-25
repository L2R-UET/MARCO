import pickle
import argparse
import json
from pathlib import Path

def extract_metrics(pkl_file):
    print(f"Extracting metrics from: {pkl_file}")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    if 'test_metrics' in data:
        print("\n" + "="*50)
        print("TEST METRICS:")
        print("="*50)
        metrics = data['test_metrics']
        
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                print(f"{key}: {value}")
        else:
            print(metrics)
        print("="*50)
        
        return metrics
    else:
        print("No 'test_metrics' found in pickle file")
        print(f"Available keys: {list(data.keys())}")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl_file', type=str, 
                       default=r"train\saved\lru\beauty\retrieved.pkl",
                       help='Path to pickle file')
    args = parser.parse_args()
    
    extract_metrics(args.pkl_file)
