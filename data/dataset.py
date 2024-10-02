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
#    2. Current version is merely an example 🪀.
#    3. I'm still unfamiliar with the NLP task paradigm, such as
#       the importance of the attention mask.
# --------------------------------------------------------------
#    Description:
#        Dataset
# --------------------------------------------------------------
#    BIG WARNING:
#        This file is just a framework and cannot be run or tes-
#        ted because 1) real news articles are unavailable, and 
#        2) fake news hasn't been generated yet.
# ==============================================================



# packages
import os

import pandas as pd

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from data._gen_data import get_df



# dataset
class News(Dataset):
    real_csv_path = "/scratch/ym2380/data/news/news.csv"
    #WARNING: Do I need to warn you that this path is fake?
    fake_csv_path = "Ironic, as the path to fake news is itself fake."
    #WARNING: This column name is fake. The real data lacks this column, and the fake data doesn't exsit.
    article_column_name = "article"
    
    def __init__(self, tokenizer, max_length):
        if not os.path.exists(self.real_csv_path):
            self._download_csv()
        if not os.path.exists(self.fake_csv_path):
            raise FileNotFoundError("Error: File not found!")
        self.real_data = pd.read_csv(self.real_csv_path)
        self.fake_data = pd.read_csv(self.fake_csv_path)

        #WARNING: using pre-tokenization would be more efficient
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        #WARNING: The dataset length depends on its structure. For instance, if each iteration yields 
        #         one real and one fake news pair, and each real news has only one fake counterpart, 
        #         the length would be len(self.real_data). However, if multiple fake news items corre-
        #         spond to one real news, or pairs are based on topic rather than the news being real 
        #         or fake, the length may vary. For now, we're setting the length to len(real_data) +
        #         len(fake_real) because I'm out of my mind 🤪.
        return len(self.real_data) + len(self.fake_data)

    def __getitem__(self, idx):
        #WARNING: Like the dataset length, the method to fetch fake news will depend on the dataset 
        #         structure, so I'll implement several approaches.
        #WARNING: The following implementations are inefficient—you already know this, so I won't elaborate.
        ikemen_kamen_amai_masuku = 1
        if ikemen_kamen_amai_masuku == 1:
            article_sun = self.real_data.loc[idx, self.article_column_name]
            hash_id_ = self.real_data.loc[idx, "hash_id"]
            article_moon = self.fake_data.loc[fake_data["hash_id"] == hash_id_, self.article_column_name].iloc[0]
        elif ikemen_kamen_amai_masuku == 11:
            #COMMENT: One number. What is happening?
            eleven = 111
            real_idx = idx // eleven
            fake_idx = idx % eleven
            article_sun = self.real_data.loc[real_idx, self.article_column_name]
            hash_id_ = self.real_data.loc[real_idx, "hash_id"]
            article_moon = self.fake_data.loc[fake_data["hash_id"] == hash_id_, self.article_column_name].iloc[fake_idx]
        elif ikemen_kamen_amai_masuku == -1:
            pass
            #COMMENT: skipped 🦘.

        #WARNING: The tokenization part deserves double-checking, as I’m not very familiar with it.
        real_encoding = self.tokenizer.encode_plus(
                real_article,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                )
        real_input_ids = real_encoding['input_ids'].squeeze()
        
        fake_encoding = self.tokenizer.encode_plus(
                fake_article,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                return_attention_mask=True,
                return_tensors="pt",
                )
        fake_input_ids = fake_encoding['input_ids'].squeeze()
        
        #WARNING: attention masks should be important
        #attention_mask = real_encoding['attention_mask'].squeeze()
        #attention_mask = fake_encoding['attention_mask'].squeeze()
        
        return (real_input_ids, fake_input_ids)
    
    def _download_csv(self):
        df = get_df(db_name="rss_reader")

        dir_ = os.path.dirname(self.real_csv_path)
        if not os.path.exists(dir_):
            os.makedirs(dir_)
        df.to_csv(self.real_csv_path, index=False)
