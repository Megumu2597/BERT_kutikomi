import os
import matplotlib.pyplot as plt
import json
from collections import defaultdict
import os
from pathlib2 import Path
import sqlite3, collections
import pandas as pd, itertools as it
from glob import glob
pd.options.display.max_rows=15
pd.options.display.max_columns=15

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from nlp import load_dataset
import sys
from tqdm.notebook import tqdm
import pickle
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from transformers import AdamW


print("load data")
data_dir = '/work/g11009/kwk/share/retty/data'
reviews = pd.read_csv(os.path.join(data_dir, 'retty_report_metrics_raw.csv'), header=None)
reviews.columns = ['user_code', 'restaurant_id', 'n_report', 'review', 'visit_time', 'report_time', 'timestamp_of_submission']

use_cols = ['user_code', 'review']
reviews = reviews[reviews.review.notnull()][use_cols]
demographics = pd.read_csv('../../u00643/data/prj_dss_genderage_fold_by_park.csv')
review_with_label = reviews.merge(demographics)
review_with_label.gender = review_with_label.gender == '男'
##genderを0/1に
review_with_label.loc[review_with_label['gender']== True, 'gender'] = 1
review_with_label.loc[review_with_label['gender']== False, 'gender'] = 0

train_data = review_with_label.rename(columns = {'fold': 'kfold'}, inplace = False)
subset_valid = train_data[train_data['kfold'] == 0].reset_index(drop=True)
subset_valid = subset_valid.loc[list(np.arange(9400)*100)].reset_index(drop=True)

subset_train = train_data[train_data['kfold'] != 0].reset_index(drop=True)
train_data = pd.concat([subset_train,subset_valid], ignore_index=True)
train_docs = subset_train["review"].tolist()
train_labels = subset_train["gender"].tolist()

print("prepare model")
model_name="cl-tohoku/bert-base-japanese-char"#'cl-tohoku/bert-base-japanese'
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
#model = BertForSequenceClassification.from_pretrained('path/to/dir') # load model
tokenizer = BertTokenizer.from_pretrained(model_name)

encodings = tokenizer(train_docs, return_tensors='pt', padding=True, truncation=True, max_length=128)
input_ids = encodings['input_ids']
attention_mask = encodings['attention_mask']

optimizer = AdamW(model.parameters(), lr=1e-5)

labels = torch.tensor(train_labels).unsqueeze(0)
outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#print(outputs,type(outputs))
if torch.cuda.device_count() >= 1:
        print('Model pushed to {} GPU(s), type {}.'.format(
            torch.cuda.device_count(), 
            torch.cuda.get_device_name(0))
        )
        model = model.cuda() 
else:
    raise ValueError('CPU training is not supported')

print("start training")
loss = outputs[0]
loss.backward()
optimizer.step()
print("save model")

result_dict = {
        'epoch':[], 
        'train_loss': [], 
        'val_loss' : [], 
        'best_val_loss': np.inf
    }
result_list = []
result_list.append(result_dict)
with open('roberta_log_list.pickle', 'wb') as handle:
    pickle.dump(result_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

model.save_pretrained(f"model_0_fold.bin") # save
