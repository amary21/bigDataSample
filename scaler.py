# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from elasticsearch import Elasticsearch, helpers, exceptions


domain = "url_elasticsearch"
port = 443

# concatenate host string from values
host = domain + ":" + str(port)
client = Elasticsearch(host)

search_body = {
    "size": 500,
        "query": {
            "match_all": {}
        }
    }

resp = client.search(
        index = "twitter",
        body = search_body,
        scroll = '3m', # time value for search
    )


def calc_median_favorites(df, kolom_text, favorite_count):
    
    fav_list = []
    
    for index, row in df.iterrows():
        # Remove retweets
        if row[kolom_text].startswith('RT'):
            continue
        # Remove replies
        elif row[kolom_text].startswith('@'):
            continue
        else:
            fav_list.append(row[favorite_count])
    
    return np.median(fav_list)


def calc_median_retweets(df, kolom_text, retweet_count):
    
    rt_list = []
    
    for index, row in df.iterrows():
        if row[kolom_text].startswith('RT'):
            continue
        else:
            rt_list.append(row[retweet_count])
    
    return np.median(rt_list) 


def remove_low_engagement_accounts(df):
    
    working_df = df.copy()
    
    working_df = working_df.loc[(working_df['median_favs'] > 0) &
                                (working_df['user_followers_count'] >= 1000)]
    
    return working_df


def create_engagement_metric(df):
      
    working_df = df.copy()
    
    from sklearn.preprocessing import MinMaxScaler
    # Favorites
    fav_eng_array = df['median_favs'] / df['user_followers_count']
    scaler = MinMaxScaler().fit(fav_eng_array.values.reshape(-1, 1))
    scaled_favs = scaler.transform(fav_eng_array.values.reshape(-1, 1))
    
    # Retweets
    rt_eng_array = df['median_favs'] / df['user_followers_count']
    scaler = MinMaxScaler().fit(rt_eng_array.values.reshape(-1, 1))
    scaled_rts = scaler.transform(rt_eng_array.values.reshape(-1, 1))
    
    mean_eng = (scaled_favs + scaled_rts) / 2
    working_df['engagement_metric'] = mean_eng
    
    return working_df


data = [x["_source"] for x in resp["hits"]["hits"]]
df_raw = pd.DataFrame(data)
df_raw = df_raw.dropna()
df_raw['median_favs'] = calc_median_favorites(df_raw, 'full_text', 'favourite_count')
df_raw['median_rts'] = calc_median_retweets(df_raw, 'full_text', 'retweet_count')
df_raw = df_raw.loc[~df_raw['median_favs'].isnull()]
df_raw = remove_low_engagement_accounts(df_raw)
df_raw = create_engagement_metric(df_raw)

fig, (ax1) = plt.subplots(ncols=1, figsize=(6, 5))
ax1.set_title('After Scaling')

sns.kdeplot(df_raw['engagement_metric'], ax=ax1)
plt.show()
print(df_raw[1:10])
normalized_df=(df_raw['engagement_metric']-df_raw['engagement_metric'].min())/(df_raw['engagement_metric'].max()-df_raw['engagement_metric'].min())
print(normalized_df)
#print(calc_median_favorites(df_raw, 'full_text', 'favourite_count'))
#print(calc_median_retweets(df_raw, 'full_text', 'retweet_count'))