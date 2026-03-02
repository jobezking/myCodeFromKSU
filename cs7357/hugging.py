from datasets import load_dataset

data_files = {
    'train': '/media/type3/data/train_dm.csv',
    'valid': '/media/type3/data/valid_dm.csv'
}

dataset = load_dataset(
    "csv",
    data_files=data_files,
    delimiter="\t"
)

train_data = dataset['train']
valid_data = dataset['valid']

print(train_data)
print(valid_data)
