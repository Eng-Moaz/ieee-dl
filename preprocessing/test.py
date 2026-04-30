from preprocessing.data_loaders import CustomData

from modeling.config import config

dataset = CustomData(path=config.raw_data_dir)

example = dataset[0]

print(example)