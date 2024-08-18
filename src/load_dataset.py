
from datasets import load_dataset, load_metric
from transformers import pipeline

def dataset_loader():
    raw_datasets = load_dataset('kde4', lang1 = 'en', lang2 = 'fr')
    return raw_datasets


def split_dataset():

    raw_datasets = dataset_loader()
    split_datasets = raw_datasets['train'].train_test_split(train_size=0.9, shuffle=True, seed=40)

    # rename the test key with validation:
    split_datasets['validation'] = split_datasets.pop('test')

    return split_datasets

