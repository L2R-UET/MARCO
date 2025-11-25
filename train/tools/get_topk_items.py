from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import torch
import argparse
from recbole.utils.case_study import full_sort_topk
from recbole.quick_start import load_data_and_model

def patch_env():
    import torch.distributed as dist
    orig_barrier = dist.barrier
    dist.barrier = lambda *a, **kw: orig_barrier(*a, **kw) if dist.is_initialized() else None
    orig_load = torch.load
    torch.load = lambda *a, **kw: orig_load(*a, **{**kw, 'map_location': torch.device('cpu')} if 'map_location' not in kw and not torch.cuda.is_available() else kw)

def get_latest_checkpoint(base_dir):
    path = os.path.join(base_dir, 'saved')
    if not os.path.exists(path):
        return None
    files = [f for f in os.listdir(path) if f.endswith('.pth')]
    if not files:
        return None
    files = sorted(files, reverse=True)
    return os.path.join(path, files[0])

def get_checkpoint_path(base_dir, pth_file=None):
    """
    Get checkpoint path. If pth_file is specified, use it.
    Otherwise, automatically select the latest checkpoint.
    """
    if pth_file:
        # If absolute path is provided
        if os.path.isabs(pth_file) and os.path.exists(pth_file):
            return pth_file
        # If relative path from saved directory
        saved_path = os.path.join(base_dir, 'saved', pth_file)
        if os.path.exists(saved_path):
            return saved_path
        # If file doesn't exist
        print(f'Checkpoint file not found: {pth_file}')
        return None
    # Auto-select latest checkpoint
    return get_latest_checkpoint(base_dir)

def load_model(checkpoint):
    if not torch.cuda.is_available():
        c = torch.load(checkpoint, map_location='cpu')
        c['config']['device'] = 'cpu'
        tmp = checkpoint + '.cpu_temp'
        torch.save(c, tmp)
        cfg, model, ds, _, _, test = load_data_and_model(tmp)
        if os.path.exists(tmp):
            os.remove(tmp)
    else:
        cfg, model, ds, _, _, test = load_data_and_model(checkpoint)
    return cfg, model, ds, test

def get_user_internal_ids(dataset):
    tokens = dataset.field2id_token[dataset.uid_field]
    return [internal_id for internal_id, token in enumerate(tokens) if token != '[PAD]' and token is not None]

def get_ground_truth_items_from_testdata(config, test_data):
    from collections import defaultdict
    uid_field = config['USER_ID_FIELD']
    iid_field = config['ITEM_ID_FIELD']
    inter_feat = test_data.dataset.inter_feat
    users = inter_feat[uid_field].cpu().numpy()
    items = inter_feat[iid_field].cpu().numpy()
    ground_truth_items = defaultdict(list)
    for u, it in zip(users, items):
        ground_truth_items[int(u)].append(int(it))
    return ground_truth_items

def prepare_model(pth_file=None):
    patch_env()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    checkpoint = get_checkpoint_path(base_dir, pth_file)
    if checkpoint is None:
        print('No checkpoint found!')
        exit(1)
    print(f'Using checkpoint: {os.path.basename(checkpoint)}')
    cfg, model, dataset, test_data = load_model(checkpoint)
    device = cfg['device']
    model.eval()
    dataset_name = getattr(cfg, 'dataset', 'unknown') if hasattr(cfg, 'dataset') else cfg['dataset'] if 'dataset' in cfg else 'unknown'
    model_name = getattr(cfg, 'model', 'unknown') if hasattr(cfg, 'model') else cfg['model'] if 'model' in cfg else 'unknown'
    user_internal_ids = get_user_internal_ids(dataset)
    ground_truth_items = get_ground_truth_items_from_testdata(cfg, test_data)
    print(f'Loaded {len(ground_truth_items)} ground truth items from test_data')
    return base_dir, model, device, user_internal_ids, ground_truth_items, dataset_name, model_name, test_data

def get_topk_results(model, device, user_internal_ids, ground_truth_items, test_data, k=20, batch_size=64):
    topk_results = []
    for i in tqdm(range(0, len(user_internal_ids), batch_size), desc='Batch'):
        batch_user_ids = user_internal_ids[i:i+batch_size]
        _, topk_idx = full_sort_topk(
            uid_series=batch_user_ids,
            model=model,
            test_data=test_data,
            k=k,
            device=device
        )
        for j, uid in enumerate(batch_user_ids):
            items = topk_idx[j].cpu().numpy()
            gt_item = ground_truth_items.get(uid, None)
            # If gt_item is a list, join as string, else keep as is
            if gt_item is None:
                gt_item_str = ''
            elif isinstance(gt_item, list):
                gt_item_str = ','.join(map(str, gt_item))
            else:
                gt_item_str = str(gt_item)
            topk_results.append({
                'user_id': uid,
                'gt_item': gt_item_str,
                **{f'item_{ix+1}': int(item) for ix, item in enumerate(items)}
            })
    return topk_results

def write_csv(topk_results, base_dir, dataset_name, model_name):
    df = pd.DataFrame(topk_results)
    output_dir = os.path.join(base_dir, 'recommendations')
    os.makedirs(output_dir, exist_ok=True)
    file = os.path.join(output_dir, f'{dataset_name}-for-{model_name}.csv')
    df.to_csv(file, index=False)
    print(f'File saved at: {file}')
    print(f'Total users: {len(topk_results)}')

def calculate_metrics(topk_results, k_list=[1, 3, 5, 10, 20]):
    metrics = {k: {'recall': [], 'ndcg': []} for k in k_list}
    
    for result in topk_results:
        gt_item_str = result['gt_item']
        if not gt_item_str:
            continue
            
        gt_items = set(map(int, gt_item_str.split(',')))
        
        rec_items = []
        for i in range(1, max(k_list) + 1):
            key = f'item_{i}'
            if key in result:
                rec_items.append(result[key])
            else:
                break
                
        for k in k_list:
            rec_k = rec_items[:k]
            hits = 0
            for item in rec_k:
                if item in gt_items:
                    hits += 1
            
            # Recall@K
            recall = hits / len(gt_items) if len(gt_items) > 0 else 0
            metrics[k]['recall'].append(recall)
            
            # NDCG@K
            dcg = 0
            idcg = 0
            for i, item in enumerate(rec_k):
                if item in gt_items:
                    dcg += 1 / np.log2(i + 2)
            
            for i in range(min(len(gt_items), k)):
                idcg += 1 / np.log2(i + 2)
                
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[k]['ndcg'].append(ndcg)
            
    print("\nMetrics for the last 2000 users:")
    for k in k_list:
        avg_recall = np.mean(metrics[k]['recall'])
        avg_ndcg = np.mean(metrics[k]['ndcg'])
        print(f"Recall@{k}: {avg_recall:.4f}")
        print(f"NDCG@{k}: {avg_ndcg:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Get top-k recommendations from trained model')
    parser.add_argument('--pth_file', type=str, default=None,
                        help='Path to checkpoint file (.pth). If not specified, automatically uses the latest checkpoint.')
    parser.add_argument('--k', type=int, default=20,
                        help='Number of top items to recommend (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference (default: 64)')
    args = parser.parse_args()
    
    base_dir, model, device, user_internal_ids, ground_truth_items, dataset_name, model_name, test_data = prepare_model(args.pth_file)
    topk_results = get_topk_results(model, device, user_internal_ids, ground_truth_items, test_data, k=args.k, batch_size=args.batch_size)
    write_csv(topk_results, base_dir, dataset_name, model_name)
    
    # Calculate metrics for the last 2000 users
    last_2000_results = topk_results[-2000:]
    calculate_metrics(last_2000_results)

if __name__ == '__main__':
    main()
