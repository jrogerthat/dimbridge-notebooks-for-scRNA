import torch
from torch import nn
from torch import optim

from flask import Flask
from flask import request
from flask_cors import CORS

import numpy as np
import pandas as pd

from textwrap import dedent
from base64 import b64encode

# from tqdm import tqdm

# create Flask app
app = Flask(__name__)
CORS(app)
device = "cuda" if torch.cuda.is_available() else "cpu"


def predict(x, a, mu):
    r"""
    UMAP-inspired predict function.
    A bump function centered at $\\mu$ with extent determined by $1/|a|$.

    $$ pred = \frac{1}{1+ \sum_{i=1}^{p} |a_i| * |x_i - \mu_i|^{b}} $$

    Parameters
    ----------
    x - Torch tensor, shape [n_data_points, n_features]
        Input data points
    a - Torch tensor, shape [n_features]
        A parameter for the bounding box extent. 1/a.abs() is the extent of bounding box at prediction=0.5
    mu - Torch tensor, shape [n_features]
        A parameter for the bounding box center
    b - Scalar.
        Hyperparameter for predict function. Power exponent

    Returns
    -------
    pred - Torch tensor of predction for each point in x, shape = [n_data_points, 1]
    """

    b = 5
    pred = 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1))
    return pred


# test: UMAP-inspired predict function
# n = 100
# x = torch.linspace(-3,3,n).view(n,1)
# a = torch.tensor(0.5)
# plt.plot(x, predict(x, a))


# def compute_predicate(x0, selected, n_iter=1000, mu_init=None, a_init=0.4):
#     '''
#         x0 - numpy array, shape=[n_points, n_feature]. Data points
#         selected - boolean array. shape=[n_points] of selection
#     '''
#     # prepare training data
#     # orginal data extent
#     n_points, n_features = x0.shape
#     vmin = x0.min(0)
#     vmax = x0.max(0)
#     x = torch.from_numpy(x0.astype(np.float32))
#     label = torch.from_numpy(selected).float()
#     # normalize
#     mean = x.mean(0)
#     scale = x.std(0) + 0.1
#     x = (x - mean) / scale
#     # Trainable parameters
#     # since data is normalized,
#     # mu can initialized around mean_pos examples
#     # a can initialized around a constant across all axes
#     center_selected = x[selected].mean(0)
#     if mu_init is None:
#         mu_init = center_selected
#     a = (a_init + 0.1*(2*torch.rand(n_features)-1))
#     mu = mu_init + 0.1 * (2*torch.rand(x.shape[1]) - 1)
#     a.requires_grad_(True)
#     mu.requires_grad_(True)
#     # weight-balance selected vs. unselected based on their size
#     n_selected = selected.sum()
#     n_unselected = n_points - n_selected
#     instance_weight = torch.ones(x.shape[0])
#     instance_weight[selected] = n_points/n_selected
#     instance_weight[~selected] = n_points/n_unselected
#     bce = nn.BCELoss(weight=instance_weight)
#     optimizer = optim.SGD([
#         {'params': mu, 'weight_decay': 0},
#         # smaller a encourages larger reach of the bounding box
#         {'params': a, 'weight_decay': 0.0}
#     ], lr=1e-2, momentum=0.9)
#     # training loop
#     for e in range(n_iter):
#         pred = predict(x, a, mu)
#         loss = bce(pred, label)
#         loss += (mu - center_selected).pow(2).mean() * 20
#         # loss += a.abs().mean() * 100
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if e % (n_iter//5) == 0:
#             # print(pred.min().item(), pred.max().item())
#             print(f'[{e:>4}] loss {loss.item()}')
#     a.detach_()
#     mu.detach_()
#     # plt.stem(a.abs().numpy()); plt.show()
# pred = (pred > 0.5).float()
# correct = (pred == label).float().sum().item()
# total = selected.shape[0]
# accuracy = correct/total
# # 1 meaning points are selected
# tp = ((pred == 1).float() * (label == 1).float()).sum().item()
# fp = ((pred == 1).float() * (label == 0).float()).sum().item()
# fn = ((pred == 0).float() * (label == 1).float()).sum().item()
# precision = tp/(tp+fp)
# recall = tp/(tp+fn)
# f1 = 1/(1/precision + 1/recall)
# print(f'''
# accuracy = {correct/total}
# precision = {precision}
# recall = {recall}
# f1 = {f1}
# ''')
# # predicate clause selection
# # r is the range of the bounding box on each dimension
# # bounding box is defined by the level set of prediction=0.5
# r = 1 / a.abs()
# predicates = []
# for k in range(mu.shape[0]):
#     # denormalize
#     r_k = (r[k] * scale[k]).item()
#     mu_k = (mu[k] * scale[k] + mean[k]).item()
#     ci = [mu_k - r_k, mu_k + r_k]
#     assert ci[0] < ci[1], 'ci[0] is not less than ci[1]'
#     if ci[0] < vmin[k]:
#         ci[0] = vmin[k]
#     if ci[1] > vmax[k]:
#         ci[1] = vmax[k]
#     # feature selection based on extent range
# #         should_include = r[k] < 1.0 * (x[:,k].max()-x[:,k].min())
#     should_include = not (ci[0] <= vmin[k] and ci[1] >= vmax[k])
#     if should_include:
#         predicates.append(dict(
#             dim=k, interval=ci
#         ))
# for p in predicates:
#     print(p)
# return predicates, mu, a, [accuracy, precision, recall, f1]


def compute_predicate_sequence(
    x0,
    selected,
    attribute_names=[],
    n_iter=1000,
    device=device,
):
    """
    x0 - numpy array, shape=[n_points, n_feature]. Data points
    selected - boolean array. shape=[brush_index, n_points] of selection
    """
    mu_init = None
    a_init = 0.4

    n_points, n_features = x0.shape
    n_brushes = selected.shape[0]

    # prepare training data
    # orginal data extent
    vmin = x0.min(0)
    vmax = x0.max(0)
    x = torch.from_numpy(x0.astype(np.float32)).to(device)
    label = torch.from_numpy(selected).float().to(device)
    # normalize
    mean = x.mean(0)
    scale = x.std(0) + 0.1
    x = (x - mean) / scale

    # Trainable parameters
    # since data is normalized,
    # mu can initialized around mean_pos examples
    # a can initialized around a constant across all axes
    selection_centroids = torch.stack([x[sel_t].mean(0) for sel_t in selected], 0)

    # initialize the bounding box center (mu) at the data centroid, +-0.1 at random
    if mu_init is None:
        mu_init = selection_centroids
    a = (a_init + 0.1 * (2 * torch.rand(n_brushes, n_features) - 1)).to(device)
    mu = mu_init + 0.1 * (2 * torch.rand(n_brushes, x.shape[1], device=device) - 1)
    a.requires_grad_(True)
    mu.requires_grad_(True)

    # For each brush,
    # weight-balance selected vs. unselected based on their size
    # and create a weighted BCE loss function (for each brush)
    bce_per_brush = []
    for st in selected:  # for each brush, define their class-balanced loss function
        n_selected = st.sum()  # st is numpy array
        n_unselected = n_points - n_selected
        instance_weight = torch.ones(x.shape[0]).to(device)
        instance_weight[st] = n_points / n_selected
        instance_weight[~st] = n_points / n_unselected
        bce = nn.BCELoss(weight=instance_weight)
        bce_per_brush.append(bce)

    optimizer = optim.SGD(
        [
            {"params": mu, "weight_decay": 0},
            # smaller a encourages larger reach of the bounding box
            {"params": a, "weight_decay": 0.01},
        ],
        lr=1e-2,
        momentum=0.9,
    )

    # training loop
    for e in range(n_iter):
        loss_per_brush = []
        for t, st in enumerate(selected):  # for each brush, compute loss
            # TODO try subsample:
            # use all selected data
            # randomly sample unselected data with similar size
            pred = predict(x, a[t], mu[t])
            loss = bce(pred, label[t])
            loss += (mu[t] - selection_centroids[t]).pow(2).mean() * 20
            loss_per_brush.append(loss)
        smoothness_loss = 100 * (a[1:] - a[:-1]).pow(2).mean()
        smoothness_loss += 100 * (mu[1:] - mu[:-1]).pow(2).mean()
        # print('bce', loss_per_brush)
        # print('smoothness', smoothness_loss.item())
        sparsity_loss = 0  # a.abs().mean() * 100
        total_loss = sum(loss_per_brush) + smoothness_loss + sparsity_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        if e % (n_iter // 5) == 0:
            # print(pred.min().item(), pred.max().item())
            print(f"[{e:>4}] loss {loss.item()}")
    a.detach_()
    mu.detach_()
    # plt.stem(a.abs().numpy()); plt.show()

    qualities = []
    for t, st in enumerate(selected):  # for each brush, compute quality
        pred = predict(x, a[t], mu[t])
        pred = (pred > 0.5).float()
        correct = (pred == label[t]).float().sum().item()
        total = n_points
        accuracy = correct / total
        # 1 meaning points are selected
        tp = ((pred == 1).float() * (label == 1).float()).sum().item()
        fp = ((pred == 1).float() * (label == 0).float()).sum().item()
        fn = ((pred == 0).float() * (label == 1).float()).sum().item()
        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1 = 2 / (1 / precision + 1 / recall) if precision > 0 and recall > 0 else 0
        print(
            dedent(
                f"""
            brush = {t}
            accuracy = {accuracy}
            precision = {precision}
            recall = {recall}
            f1 = {f1}
        """
            )
        )
        qualities.append(
            dict(brush=t, accuracy=accuracy, precision=precision, recall=recall, f1=f1)
        )

    # predicate clause selection
    # r is the range of the bounding box on each dimension
    # bounding box is defined by the level set of prediction=0.5
    predicates = []
    # for each brush, generate a predicate from a[t] and mu[t]
    for t, st in enumerate(selected):
        r = 1 / a[t].abs()
        predicate_clauses = []
        for k in range(n_features):  # for each attribute
            vmin_selected = x0[st, k].min()
            vmax_selected = x0[st, k].max()
            # denormalize
            r_k = (r[k] * scale[k]).item()
            mu_k = (mu[t, k] * scale[k] + mean[k]).item()
            ci = [mu_k - r_k, mu_k + r_k]
            assert ci[0] < ci[1], "ci[0] is not less than ci[1]"
            if ci[0] < vmin[k]:
                ci[0] = vmin[k]
            if ci[1] > vmax[k]:
                ci[1] = vmax[k]

            # feature selection based on extent range
            #         should_include = r[k] < 1.0 * (x[:,k].max()-x[:,k].min())
            should_include = not (ci[0] <= vmin[k] and ci[1] >= vmax[k])

            if should_include:
                if ci[0] < vmin_selected:
                    ci[0] = vmin_selected
                if ci[1] > vmax_selected:
                    ci[1] = vmax_selected
                predicate_clauses.append(
                    dict(
                        dim=k,
                        interval=ci,
                        attribute=columns[k],
                    )
                )
        predicates.append(predicate_clauses)
    parameters = dict(mu=mu, a=a)
    return predicates, qualities, parameters


def load_data(dataset):
    """dataset - name of the file on disk"""
    df = pd.read_csv(f"./dataset/{dataset}.csv")
    # drop certain columns if needed
    # TODO should be done without hard coding
    for attr in [
        "x",  # all
        "y",  # all
        "image_filename",  # animals
        "replication",  # for gaits
        "gene",  # for genes
        "id",  # for genes
    ]:
        if attr in df.columns:
            df = df.drop(attr, axis="columns")
    if dataset == "gait1":
        df = df[::6, :]
    x0 = df.to_numpy()
    columns = df.columns
    return x0, columns


current_dataset = None
x0 = None
columns = None


@app.route("/get_predicates", methods=["POST"])
def get_predicate():
    global current_dataset, x0, columns
    dataset = request.json["dataset"]
    # load dataset csv
    if current_dataset != dataset:
        x0, columns = load_data(dataset)
        current_dataset = dataset
    # a sequence of bool arrays indexed by [brush time, data point index]
    subsets = np.array(request.json["subsets"])
    print(x0.shape)

    # # Option 1: each brush is indepedently optimized through compute_predicate()
    # predicates = []
    # qualities = []
    # mu = None
    # a = 0.4
    # for t, subset in enumerate(subsets):
    #     predicate, mu, a, [accuracy, precision, recall, f1] = compute_predicate(
    #         x0, subset, mu_init=mu, a_init=a)
    #     predicates.append(predicate)
    #     qualities.append(dict(
    #         brush=t,
    #         accuracy=accuracy,
    #         precision=precision,
    #         recall=recall,
    #         f1=f1,
    #     ))

    # Option 2: jointly optimize the sequence
    predicates, qualities, parameters = compute_predicate_sequence(
        x0, subsets, attribute_names=columns, n_iter=1000
    )

    return dict(
        predicates=predicates,
        qualities=qualities,
    )


# embedding = None
# @app.route('/get_embedding', methods=['GET'])
# def get_embedding():
#     return {
#         'shape': embedding.shape,
#         'value': b64encode(embedding.astype(np.float32).tobytes()).decode()
#     }


# df = pd.read_csv('./dataset/gait_joined.csv')
# x0 = df.to_numpy()

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     '--embedding_fn',
    #     required=True,
    #     help='embedding file')
    # opt = parser.parse_args()
    # print(opt)
    # embedding = np.load(opt.embedding_fn)

    app.run(host="0.0.0.0", port=9001, debug=False)