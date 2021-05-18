from fastapi import FastAPI, Request
import pickle
import pandas as pd
import numpy as np
from util import *
import warnings
warnings.filterwarnings('ignore')


def get_path(clf, x_input):
    features_list = (["Manufacturing Region"]*3 +
                 ["Intercompany"]*2 +
                 ["Customer industry"]*15 +
                 ["Customer Region"]*3 +
                 ["Top Customer Group"]*2 +
                 ["Day of week"]*2 + 
                 ["Part of month"]*2 +
                 ["Part of year"]*2 +
                 ["Ordered qty"] +
                 ["Invoiced price"] + 
                 ["Cost of part"] + 
                 ['GM%']+
                 ["'# of unique products on a quote'"])
    # INIT
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    
    used_features = set()

    node_depth = np.zeros(shape=n_nodes, dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, 0)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, depth = stack.pop()
        node_depth[node_id] = depth

        # If the left and right child of a node is not the same we have a split
        # node
        is_split_node = children_left[node_id] != children_right[node_id]
        # If a split node, append left and right children and depth to `stack`
        # so we can loop through them
        if is_split_node:
            stack.append((children_left[node_id], depth + 1))
            stack.append((children_right[node_id], depth + 1))
        else:
            is_leaves[node_id] = True

    # EVALUATING
    node_indicator = clf.decision_path(x_input)
    leaf_id = clf.apply(x_input)

    sample_id = 0
    # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
    node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

    for node_id in node_index:
        # continue to the next node if it is a leaf node
        if leaf_id[sample_id] == node_id:
            continue

        # check if value of the split feature for sample 0 is below threshold
        if (x_input[sample_id, feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

#         print("decision node {node} : (X_test[{sample}, {feature}] = {value}) "
#               "{inequality} {threshold})".format(
#                   node=node_id,
#                   sample=sample_id,
#                   feature=feature[node_id],
#                   value=x_input[sample_id, feature[node_id]],
#                   inequality=threshold_sign,
#                   threshold=threshold[node_id]))
        feature_id = feature[node_id]
        used_features |= {features_list[feature_id]}
    return list(used_features)

with open('model.pkl', 'rb') as file:
    pipeline = pickle.load(file)
features = pipeline
model = pipeline.steps.pop(-1)[1]
with open('price_bands.pkl', 'rb') as file:
    price_bands = pickle.load(file)

app = FastAPI()

@app.get("/predict")
async def inference(request: Request):
    json = await request.json()
    responses = []
    if isinstance(json, dict):
        json = [json]
    for json_dict in json:
        df = pd.DataFrame(json_dict, index=[0])
        #print(df)
        df_features = features.transform(df)
        cluster = model.predict(df_features).item()
        path = get_path(model, df_features.values)
        pricing_bands = price_bands[cluster]
        price = json_dict['Invoiced price']
        responses.append(
            {
                "cluster-id": cluster,
                "important_features": path,
                "pricing": dict((letter, price * (1 + pricing_bands[letter.lower() + "_center"])) for letter in "ABCDF")
            }
        )
        #except:
        #    responses.append({"error": "There was an error with this input. Sorry for the inconvenience."})
    return responses if len(responses) > 1 else responses[0]
