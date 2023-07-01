# this file helps initialize an S3DG model with the pretrained weights
# and then start using it

import torch as th
import torch.nn as nn

from .s3dg import S3DG
from config import reader

cfg = reader()
num_classes = cfg["s3d"]["num_classes"]
s3d_model_path = cfg["s3d"]["model_path"]
s3d_classes_names = cfg["s3d"]["classes_names"]


def init_weights(model: nn.Module, state_dict: dict, should_omit="s3dg."):
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        # change the key name to match the model
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    # remove those old keys from the model state_dict
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(model, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # modify the old keys by removing a prefix from them
    if should_omit is not None:
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            if key.find(should_omit) == 0:
                old_keys.append(key)
                new_key = key[len(should_omit):]
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

    # recursively load the module and its submodules
    def load(module, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    load(model, prefix="")

    if len(missing_keys) > 0:
        print(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, "\n   " + "\n   ".join(missing_keys)
            )
        )
    if len(unexpected_keys) > 0:
        print(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, "\n   " + "\n   ".join(unexpected_keys)
            )
        )
    if len(error_msgs) > 0:
        print(
            "Weights from pretrained model cause errors in {}: {}".format(
                model.__class__.__name__, "\n   " + "\n   ".join(error_msgs)
            )
        )

    return model


def get_model():
    print("Loading S3DG ...")
    model = S3DG(num_classes)
    model = model.cuda()
    model_data = th.load(s3d_model_path)
    model = init_weights(model, model_data)


