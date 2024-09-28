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
#                 FILE: ./data/_utils.py
# --------------------------------------------------------------
#                          TODOs:
#    1. Split the data
# --------------------------------------------------------------
#    Description:
#        Implement basic methods of data, e.g. get datasets & g-
#        et dataloaders
# --------------------------------------------------------------
# ==============================================================



# packages
import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer

from data.dataset import News



#WARNING: get_data and get_loader are surprisingly simple, which might be a concern

# get data
def get_data(args):
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    data = News(tokenizer=tokenizer, max_length=args.tokenizer_max_length)

    return data



# get loader
def get_loader(args, data):
    loader = DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            )
    return loader
