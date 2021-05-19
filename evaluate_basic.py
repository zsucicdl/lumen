import pickle
import time

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
import json
import requests

data_path = "LUMEN0.pkl"
with open(data_path, 'rb') as f:
    df = pickle.load(f)

df = df[~df['GM%'].isnull()]
df = df[df['GM%'] > 0]
df = df[df['GM%'] <= 1]

# df[df['GM%'] > 1]['GM%'] = 1
# df[df['GM%'] < -1]['GM%'] = -1

df = df[df['Invoiced qty (shipped)'] > 0]
df = df[df['Ordered qty'] > 0]
df = df[df['Invoiced price'] > 0]
df = df[df['Cost of part'] > 0]

df = df[~df['# of unique products on a quote'].isnull()]
df = df[df['# of unique products on a quote'] < 374]
df = df[df['# of unique products on a quote'] > 0]

df = df[~df['Product group'].isnull()]
df = df[~df['Manufacturing Location Code'].isnull()]
df = df[~df['Make vs Buy'].isnull()]
df = df[~df['Customer industry'].isnull()]
df = df[~df['Customer Region'].isnull()]

# df['Invoiced qty (shipped)'].astype(int)
# df['Ordered qty'].astype(int)
# df['# of unique products on a quote'].astype(int)


le=100

data0 = df[0:le]
# data={k:v[1] for row in data0 for k,v in row.items()}

for i in range(le):
    data = {k: list(v.values())[0] for k, v in data0[i:i+1].to_dict().items()}
    data_json = json.dumps(data)
    payload = {'json_payload': data_json}#, 'apikey': 'YOUR_API_KEY_HERE'}
    r = requests.get('http://127.0.0.1:8000/predict', data=data_json)
    #r = requests.get('http://127.0.0.1:8000', data=payload)
    print(r.json())

# preds = clf.predict(x_valid)
# print(accuracy_score(y_valid, preds))
# print(r2_score(y_valid, preds))
#
# ind_in = ((cdf['GM%'] <= cdf['A']) & (cdf['GM%'] >= cdf['F']))
# print(sum(ind_in) / cdf.shape[0])
# rmse_gma = np.sqrt(mean_squared_error(cdf['GM%'], cdf['A']))
# print(rmse_gma)
# avg_range = (cdf['A'].sum() - cdf['F'].sum()) / cdf.shape[0]
# print(avg_range)

# ind_wrong = (preds != y_valid)
# sv_report = sv.analyze(x_valid[ind_wrong])
# sv_report.show_html('SV_report_wrongcluster_x.html')
