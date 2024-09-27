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
#    1. Implement saving of generated data points
# --------------------------------------------------------------
#    Description:
#        Generate augmented data points in advance
# --------------------------------------------------------------
# ==============================================================



# packages
import argparse

import pandas

from transformers import pipeline

from data import News
from utils import get_datetime_filename



# args
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_option", default="text-generation")
    parser.add_argument("--model_name", default="gpt-3.5-turbo")

    parser.add_argument("--num_augmentations_per_news", type=int, default=6)
    parser.add_argument("--max_length_per_generated", type=int, default=512)

    args = parser.parse_args()
    return args



# generate fake news of a single news
def gen_single_augmentations(
        prototype,
        generator,
        prompt,
        #WARNING: not sure if max_length == 512 is appropriate
        max_length=512,
        num_augmentations=6):
    prompt = prompt.format(prototype)

    augmenteds = []
    raw_augmenteds = generator(prompt, max_length=max_length, num_return_sequences=num_augmentations)
    for augmented in raw_augmenteds:
        augmenteds.append(augmented["generated_text"])
    
    return augmenteds



# generate fake news of a batch of news (.csv)
def gen_batch_augmentations(
        args,
        csv_path,
        prototype_column_name,
        generator,
        prompt,
        id_column_name="hash_id",
        ):

    #WARNING: don't want to handle this exception
    if not os.path.exists(csv_path):
        raise FileNotFoundError
    
    data_df = pd.read_csv(csv_path)
    rows = list()
    #WARNING: didn't bother handling the titles
    for (row_idx, prototype) in enumerate(data_df[prototype_column_name]):
        augmenteds = gen_single_augmentations(
                prototype=prototype,
                generator=generator,
                prompt=prompt,
                max_length=args.max_length_per_generated,
                num_augmentations=args.num_augmentations_per_news,
                )
        rows.extend([(data_df.loc[row_idx, id_column_name], augmented) for augmented in augmenteds])
    
    #WARNING: naming the columns of articles to articles
    df = pd.DataFrame(rows, columns=["hash_id", "articles"])
    return df



# main
if __name__ == "__main__":
    args = get_args()

    #WARNING: not sure if the task option being "text-generation" is the best
    #WARNING: not sure having access to "gpt-3.5-turbo" in an unlimited way
    generator = pipeline(args.task_option, model=args.model_name)
    #WARNING: This prompt is topic-based, but we can also use article-based prompts.
    #WARNING: Should finetune the prompt
    prompt = f"Generate a news article based on this topic: {}"

    csv_path = News.csv_path
    #WARNING: be mindful that this columns stores descriptions currently
    prototype_column_name = News.article_column_name

    #WARNING: storing data in pd.DataFrame
    df = gen_batch_augmentations(
            args=args,
            csv_path = csv_path,
            prototype_column_name=prototype_column_name,
            generator=generator,
            prompt=prompt,
            )

    filename = get_datetime_filename(filename_fmt="d{}.csv")
    file_path = os.path.join(os.path.dirname(csv_path), filename)
    df.to_csv(file_path)
