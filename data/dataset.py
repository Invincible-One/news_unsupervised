# ==============================================================
#                      _            _             _      
#                     /\ \         /\ \     _    /\ \    
#                    /  \ \       /  \ \   /\_\ /  \ \   
#                   / /\ \ \     / /\ \ \_/ / // /\ \ \  
#                  / / /\ \ \   / / /\ \___/ // / /\ \_\ 
#                 / / /  \ \_\ / / /  \/____// /_/_ \/_/ 
#                / / /   / / // / /    / / // /____/\    
#               / / /   / / // / /    / / // /\____\/    
#              / / /___/ / // / /    / / // / /______    
#             / / /____\/ // / /    / / // / /_______\   
#             \/_________/ \/_/     \/_/ \/__________/   
#                                          
# --------------------------------------------------------------
#                 Project: News Unsupervised
#                 Author: ONE
# --------------------------------------------------------------
#                 FILE: ./pre_augment.py
# --------------------------------------------------------------
#                          TODOs:
#    1. Split the dataset.
#    2. Get articles in the database, instead of descriptions
#    3. Current version is merely an example, actual implementa-
#       tion requiring false news fetching and tuple pair items.
# --------------------------------------------------------------
#    Description:
#        Dataset
# --------------------------------------------------------------
# ==============================================================



# packages
import os

import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import Dataloader

from data._gen_data import get_df



# dataset
class News(Dataset):
    csv_path = "/scratch/ym2380/data/news/news.csv"
    #WARNING: currently this column contains descriptions
    article_column_name = "description"
    
    def __init__(self, tokenizer, max_length):
        if not os.path.exists(self.csv_path):
            self._download_csv()
        self.data = pd.read_csv(self.csv_path)

        #WARNING: using pre-tokenization would be more efficient
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = str(self.data[self.article_column_name][idx])

        #WARNING: The tokenization part deserves double-checking, as I’m not very familiar with it.
        encoding = self.tokenizer.encode_plus(
                article,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        return (input_ids, attention_mask)
    
    def _download_csv(self):
        df = get_df(db_name="rss_reader")

        _csv_dir = os.path.dirname(self.csv_path)
        if not os.path.exists(_csv_dir):
            os.makedirs(_csv_dir)
        df.to_csv(self.csv_file, index=False)
