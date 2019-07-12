# Copyright 2019 Astex Therapeutics Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import

import numpy as np
import pickle


def normalize(a, skip_norm_mask=None, params_pickle=None, params_dict=None):
    """ normalizez a 3d array (mol, atom, props) in prop axis """

    a_2d = np.reshape(a, (a.shape[0] * a.shape[1], a.shape[2]))
    if params_pickle is not None:
        with open(params_pickle) as f:
            d = pickle.load(f)
            mean = d["mean"]
            std = d["std"]
    elif params_dict is not None:
        mean = params_dict["mean"]
        std = params_dict["std"]
    else:
        mean = np.nanmean(a_2d, axis=0, keepdims=True)
        std = np.nanmean(a_2d, axis=0, keepdims=True)

    if skip_norm_mask is not None:
        # do not standardized one-hot-encoded properties
        skip_norm_mask = np.expand_dims(skip_norm_mask, 0)
        # do not standardize all-zero props
        skip_norm_mask |= (std == 0.0)

        mean[skip_norm_mask] = 0.0
        std[skip_norm_mask] = 1.0

    return np.reshape((a_2d - mean) / std, a.shape), mean, std


def ds_to_data(ds, normalize_x=True, norm_params_infile=None,
               norm_param_outfile=None):
    """ Converts xarray dataset to numpy arrays """
    x = ds.X.values

    train_mask = ds.train.values | ds.valid.values

    if normalize_x:
        skip_norm_mask = np.array([v.startswith("is_")
                                   for v in ds.feature.values])
        x, mean, std = normalize(
            x, skip_norm_mask=skip_norm_mask, params_pickle=norm_params_infile)
        if norm_param_outfile:
            with open(norm_param_outfile, "wb") as f:
                pickle.dump(dict(mean=mean, std=std), f)
    x[np.isnan(x)] = 0.0

    y = ds.Y.values

    d = ds.D.values
    d[np.isnan(d)] = 0.0

    return x, d, y, train_mask
