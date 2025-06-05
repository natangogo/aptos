# Import necessary libraries
from torch.utils.data import DataLoader
from aptos_dataset import AptosDataset

video_dir = 'dataset/videos'
ann_dir = 'dataset/annotations/APTOS_train-val_annotation.csv'
# datasets and loaders
train_ds = AptosDataset(video_dir, ann_dir, split='train', batch_size=16)
train_loader = DataLoader(train_ds, batch_size=None)

val_ds = AptosDataset(video_dir, ann_dir, split='val', batch_size=16)
val_loader = DataLoader(val_ds, batch_size=None)