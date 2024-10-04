import pandas as pd
import numpy as np
import csv


# Load Event2018
columns = [
    "tweet_id",
    "user_name",
    "text",
    "time",
    "event_id",
    "user_mentions",
    "hashtags",
    "urls",
    "words",
    "created_at",
    "filtered_words",
    "entities",
    "sampled_words",
]
df_np = np.load("raw_data/Event2018/french_tweets.npy", allow_pickle=True)
df_2018 = pd.DataFrame(data=df_np, columns=columns)

# Load Event2012
p_part1 = "raw_data/Event2012/68841_tweets_multiclasses_filtered_0722_part1.npy"
p_part2 = "raw_data/Event2012/68841_tweets_multiclasses_filtered_0722_part2.npy"
columns = [
    "event_id",
    "tweet_id",
    "text",
    "user_id",
    "created_at",
    "user_loc",
    "place_type",
    "place_full_name",
    "place_country_code",
    "hashtags",
    "user_mentions",
    "image_urls",
    "entities",
    "words",
    "filtered_words",
    "sampled_words",
]
df_np_part1 = np.load(p_part1, allow_pickle=True)
df_np_part2 = np.load(p_part2, allow_pickle=True)
df_np = np.concatenate((df_np_part1, df_np_part2), axis=0)
df_2012 = pd.DataFrame(data=df_np, columns=columns)

# Write csv version of Event2018
df_2018[["tweet_id", "text", "created_at", "event_id"]].rename(
    columns={"tweet_id": "id", "event_id": "label"}
).to_csv("event2018.tsv", sep="\t", quoting=csv.QUOTE_ALL, index=False)

# Write csv version of Event2012
df_2012[["tweet_id", "text", "created_at", "event_id"]].rename(
    columns={"tweet_id": "id", "event_id": "label"}
).to_csv("event2012.tsv", sep="\t", quoting=csv.QUOTE_ALL, index=False)

print("Save preprocessed files to event2012.tsv and event2018.tsv")
