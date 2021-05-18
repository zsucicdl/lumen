import os
import pickle
import time

import faiss
import numpy as np
import pandas as pd
from faiss import MatrixStats
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle
import time

import sklearn
from sklearn import model_selection
import sweetviz as sv
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
from sklearn import tree

data_path = "LUMEN0.pkl"
with open(data_path, 'rb') as f:
    df = pickle.load(f)

df = df[~df['GM%'].isnull()]
df = df[df['GM%'] >= -1]
df = df[df['GM%'] <= 1]

df = df[df['Invoiced qty (shipped)'] > 0]
df = df[df['Invoiced qty (shipped)'] < 1000000]
df = df[df['Ordered qty'] > 0]
df = df[df['Ordered qty'] < 1000000]
df = df[df['Invoiced price'] > 0]
df = df[df['Invoiced price'] < 100000]
df = df[df['Cost of part'] > 0]
df = df[df['Cost of part'] < 75000]

df = df[~df['# of unique products on a quote'].isnull()]
df = df[df['# of unique products on a quote'] < 374]
df = df[df['# of unique products on a quote'] > 0]

df = df[~df['Product group'].isnull()]
df = df[~df['Manufacturing Location Code'].isnull()]
df = df[~df['Make vs Buy'].isnull()]
df = df[~df['Customer industry'].isnull()]
df = df[~df['Customer Region'].isnull()]

# df = df[df['Product family'] != 'PC010']
# df = df[df['Product family'] != 'PC001']
# df = df[df['Product family'] != 'PC016']
#df = df[df['Product family'] != 'PF000']

df['Invoiced qty (shipped)'].astype(int)
df['Ordered qty'].astype(int)
df['# of unique products on a quote'].astype(int)

deal_features=['CustomerID', 'Product group', 'Manufacturing Location Code']
df['total_part_cost']=df['Invoiced qty (shipped)']*df['Cost of part']
df['deal_size_part'] = df.groupby(deal_features)['total_part_cost'].transform('sum')

for feature_name in ['Invoiced qty (shipped)', 'Ordered qty', 'Cost of part','total_part_cost','deal_size_part']:
    feature = np.log(df[feature_name])
    feature = (feature - feature.mean()) / feature.std()
    df[feature_name] = feature


categoricals = ['Manufacturing Region', 'Manufacturing Location Code', 'Intercompany',
                'Customer industry', 'Customer Region', 'Top Customer Group',
                'Product family', 'Product group']

lencdict={name:None for name in categoricals}


#df['total_price']=df['Invoiced qty (shipped)']*df['Invoiced price']
#df['deal_size_price'] = df.groupby(deal_features)['total_price'].transform('sum')
#df['deal_gm'] = (df['deal_size_price'] - df['deal_size_part']) / df['deal_size_price']

for name in categoricals:
    lenc = LabelEncoder()
    #df[name] = (df[name].reshape(-1, 1).apply(lenc.fit_transform))
    df[name] = lenc.fit_transform(df[name])
    lencdict[name]=lenc

scaler = MinMaxScaler()
df[categoricals] = scaler.fit_transform(df[categoricals])




# df['Invoiced qty (shipped)'] = np.log10(df['Invoiced qty (shipped)'] + 1)
# df['Ordered qty'] = np.log10(df['Ordered qty'] + 1)
# df['Cost of part'] = np.log10(df['Cost of part'] + 1)
# df['Invoiced price'] = np.log10(df['Invoiced price'] + 1)#????????????????????????????????????????????????
# df['# of unique products on a quote'] = np.log10(df['# of unique products on a quote'] + 1)


print(df.shape)

# df = df[0:100000]

cldf = df[['Manufacturing Region', 'Manufacturing Location Code', 'Intercompany',
          'Customer industry', 'Customer Region', 'Top Customer Group',
          'Product family', 'Product group', 'Invoiced qty (shipped)', 'Ordered qty', 'Cost of part','total_part_cost','deal_size_part', 'GM%']]

arr = (cldf.values).copy(order='C').astype(np.float32)
ncentroids = 2000
n_init = 10
max_iter = 8
verbose = True

dim = cldf.shape[1]
print(MatrixStats(arr).comments)

start = time.time()
kmeans = faiss.Kmeans(gpu=False, d=dim, k=ncentroids, niter=max_iter, verbose=verbose)
kmeans.train(arr)
D, I = kmeans.index.search(arr, 1)
end = time.time()
print('Classifier time: ', end - start)

df['labels'] = I
df['l2'] = D


df['F'] = df.groupby('labels')['GM%'].transform('min')
df['A'] = df.groupby('labels')['GM%'].transform('max')
df['dist'] = (df['A'] - df['F']) / 4
df['D'] = df['F'] + 1 * df['dist']
df['C'] = df['F'] + 2 * df['dist']
df['B'] = df['F'] + 3 * df['dist']

# df = pd.concat([df[['Item Code', 'CustomerID']], df], axis=1)
df = df[['Manufacturing Region', 'Manufacturing Location Code', 'Intercompany',
         'Customer industry', 'Customer Region', 'Top Customer Group',
         'Product family', 'Product group', 'Invoiced qty (shipped)', 'Ordered qty', 'Cost of part','total_part_cost','deal_size_part',
         'Item Code', 'CustomerID', 'labels','l2','GM%', 'A', 'B', 'C', 'D', 'F']]



print(df.columns)

feats=['Manufacturing Region', 'Manufacturing Location Code', 'Intercompany',
         'Customer industry', 'Customer Region', 'Top Customer Group',
         'Product family', 'Product group', 'Invoiced qty (shipped)', 'Ordered qty', 'Cost of part','total_part_cost','deal_size_part']

x_train, x_valid, y_train, y_valid = model_selection \
    .train_test_split(df[feats], df['labels'], test_size=0.25)

start = time.time()
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
end = time.time()
print('Decision tree time: ', end - start)
print(clf.feature_importances_)

# preds = clf.predict(x_train)
# print(accuracy_score(y_train,preds))
# print(r2_score(y_train,preds))

preds = clf.predict(x_valid)
print(accuracy_score(y_valid,preds))
print(r2_score(y_valid,preds))

rmse_gma=np.sqrt(mean_squared_error(df['GM%'],df['A']))
print(rmse_gma)
avg_range=(df['A'].sum()-df['F'].sum())/df.shape[0]
print(avg_range)

# categoricals = ['Manufacturing Region', 'Manufacturing Location Code', 'Intercompany',
#                 'Customer industry', 'Customer Region', 'Top Customer Group',
#                 'Product family', 'Product group']
# df[categoricals] = (df[categoricals].apply(lenc.inverse_transform))
#tree.plot_tree(clf)

exit(0)
ind_wrong = (preds != y_valid)
sv_report = sv.analyze(x_valid[ind_wrong])
sv_report.show_html('SV_report_wrongcluster_x.html')

# features = ['labels']
# print(df.shape)
# grouped_df = df.groupby(features)
# with open('by_labels_naivecats_20its.txt', 'w') as file:
#     for key, item in grouped_df:
#         file.write(grouped_df.get_group(key).to_string())
#         file.write("\n\n")

