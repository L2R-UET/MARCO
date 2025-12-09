#! /bin/bash

echo Preprocessing ml-100k dataset...
python main.py --main Preprocess --data_dir data/ml-100k --dataset ml-100k --n_neg_items 7

echo Preprocessing Amazon Beauty dataset...
python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category Beauty --n_neg_items 7

# echo Preprocessing Amazon dataset...
# python main.py --main Preprocess --data_dir data --dataset amazon --amazon_category {dataset_name} --n_neg_items 7

echo Preprocessing Yelp2020 dataset...
python main.py --main Preprocess --data_dir data/Yelp2020 --dataset yelp2020 --n_neg_items 7

echo Preprocessing completed!
