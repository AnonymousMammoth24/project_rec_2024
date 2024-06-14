import json
import pandas as pd
import gzip
import numpy as np
import torch.utils.data
import re

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield json.loads(l)

def getDF(path):
    # i = 1
    # asin = {}
    # title = {}
    map_title = {}

    for d in parse(path):
        if 'title' not in d.keys():
            continue
        # asin[i] = d['asin']
        # title[i] = d['title']
        # i += 1
        map_title[d['asin']] = d['title']
    return map_title

# asin, title = getDF('../amazon-reviews/Beauty/meta_All_Beauty.json.gz')
# asin_r = {y: x for x, y in asin.items()}
# colnames = ["itemId", "userId", "Rating", "timestamp"]
# data = pd.read_csv('../amazon-reviews/Beauty/All_Beauty.csv', names=colnames)
# data['itemId'] = data['itemId'].replace(asin_r)
#
# userid = {}
# i = 1
# for u in data['userId'].values:
#     if u not in userid.keys():
#         userid[u] = i
#         i += 1


class AmazonDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, meta_path):
        # load metadata to extract product titles
        id_moviename_map = getDF(meta_path)
        uniqe_itemid = list(id_moviename_map.keys())
        colnames = ["itemId", "userId", "Rating", "timestamp"]
        data = pd.read_csv(dataset_path, names=colnames)

        data = data.drop_duplicates(subset=['itemId', 'userId'], keep=False)
        data.reset_index(drop=True, inplace=True)

        # remove itemsId that are not in unique_itemid
        data = data[data['itemId'].isin(uniqe_itemid)]

        # remove users with less than 5 ratings
        userID_value_counts = data['userId'].value_counts()
        data = data[data['userId'].isin(userID_value_counts[userID_value_counts >= 5].index)]
        data.reset_index(drop=True, inplace=True)

        # encode item ID to integers, whichever appears first gets 1, second gets 2, etc.
        itemid_map = {}
        encode_moviename_map = {}
        i = 1
        for asin in data['itemId'].values:
            if asin not in itemid_map.keys():
                itemid_map[asin] = i
                encode_moviename_map[i] = id_moviename_map[asin]
                i += 1
        data['itemId'] = data['itemId'].map(itemid_map, na_action='ignore')

        # encode user ID to integers, whichever appears first gets 1, second gets 2, etc.
        userid_map = {}
        i = 1
        for u in data['userId'].values:
            if u not in userid_map.keys():
                userid_map[u] = i
                i += 1
        data['userId'] = data['userId'].map(userid_map, na_action='ignore')

        # col_order = ["userId", "itemId", "Rating", "timestamp"]
        # data = data[col_order]

        self.items = data[["userId", "itemId"]].values.astype(np.int32)
        self.targets = self.__preprocess_target(data["Rating"].values).astype(np.float32)
        self.time = data["timestamp"].values.astype(np.int32)
        self.m_data = encode_moviename_map

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, index):
        return self.items[index], self.targets[index], self.m_data[
            self.items[index][1]]

    def __preprocess_target(self, target):
        target[target <= 3] = 0
        target[target > 3] = 1
        return target

# data=AmazonDataset('../amazon-reviews/Appliance/Appliances.csv', '../amazon-reviews/Appliance/meta_Appliances.json.gz')