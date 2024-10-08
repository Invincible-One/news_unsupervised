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
#                 FILE: ./data/_gen_data.py
# --------------------------------------------------------------
#                          TODOs:
#    1. Add dataset description printing
# --------------------------------------------------------------
#    Description:
#        Get data from databases.
# --------------------------------------------------------------
# ==============================================================



# packages & configurations
import os
import sys

import pandas as pd

import psycopg2
from sqlalchemy import create_engine
from urllib.parse import quote_plus

from utils import hash_encoder



_db_meta = {
    "HOST":"150.230.113.32",
    "PORT":"8899",
    "USER": "yima",
    "PASSWORD": "yima@09012024",
    "DB_TABLES": {
        "rss_embedding": ["news_2024",],
        "rss_reader": [
            "z_rss_t_bbc_world",
            "z_rss_t_google",
            "z_rss_t_washingtonpost_world",
            ],
    },
}



root = "/scratch/ym2380/data/news"



# get dataframe helper
def get_df(db_name):
    assert db_name in _db_meta["DB_TABLES"], f"Don't recognise {db_name}!"

    encoded_password = quote_plus(_db_meta["PASSWORD"])
    db_url = f"postgresql://{_db_meta['USER']}:{encoded_password}@{_db_meta['HOST']}:{_db_meta['PORT']}/{db_name}"
    engine = create_engine(db_url)

    dfs = list()
    #COMMNET: Original table has a typo.
    required_columns = ["title", "description", "key_infomation"]
    for table_name in _db_meta["DB_TABLES"][db_name]:
        query = f"SELECT * FROM public.{table_name}"
        df = pd.read_sql(query, engine)

        try:
            df = df[required_columns]
        except KeyError as e:
            missing_columns = list(e.args[0])
            raise KeyError(f"Required columns are missing: {missing_columns}") 
        df.rename(columns={"key_infomation": "key_information"}, inplace=True)

        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    #COMMENT: add a column of hash identifiers
    #WARNING: encoding the descriptions, not articles
    df["hash_id"] = df["description"].apply(hash_encoder)
    #WARNING: temporally commented over
    #assert df["hash_id"].nunique() == len(df), "Error: Identical hash ids detected!"

    return df
