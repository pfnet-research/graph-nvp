import json
import os

import numpy as np
from tabulate import tabulate


class Hyperparameters:
    def __init__(self, num_nodes=-1, num_relations=-1, num_features=-1, masks=None, path=None,
                 num_masks=None, mask_size=None, num_coupling=None, batch_norm=False,
                 additive_transformations=False, learn_dist=True,
                 squeeze_adj=False, prior_adj_var=1.0, prior_x_var=1.0,
                 mlp_channels=None, gnn_channels=None, seed=1):
        self.gnn_channels = gnn_channels
        self.mlp_channels = mlp_channels
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.num_features = num_features
        self.masks = masks
        self.num_masks = num_masks
        self.mask_size = mask_size
        self.num_coupling = num_coupling
        self.path = path
        self.apply_batch_norm = batch_norm
        self.additive_transformations = additive_transformations
        self.learn_dist = learn_dist
        self.squeeze_adj = squeeze_adj
        self.prior_adj_var = prior_adj_var
        self.prior_x_var = prior_x_var
        self.seed = seed

        if path is not None:
            if os.path.exists(path) and os.path.isfile(path):
                with open(path, "r") as f:
                    obj = json.load(f)
                    for (key, value) in obj.items():
                        if key == 'masks':
                            masks = dict()
                            for k in value.keys():
                                if value[k]:
                                    dtype = np.bool
                                    masks[k] = [np.array(item, dtype=np.bool) for item in value[k]]
                            value = masks
                        setattr(self, key, value)
            else:
                raise Exception("{} does not exist".format(path))

    def save(self, path):
        self.path = path
        with open(path, "w") as f:
            json.dump(self.__dict__, f, indent=4, sort_keys=True, cls=NumpyEncoder)

    def print(self):
        rows = []
        for key, value in self.__dict__.items():
            rows.append([key, value])
        print(tabulate(rows))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

