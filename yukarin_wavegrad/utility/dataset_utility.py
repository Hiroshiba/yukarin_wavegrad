# https://github.com/pytorch/pytorch/blob/b522a8e1ff8a531c4ac75a3551b99d5b40125cf0/torch/utils/data/_utils/collate.py

import re

import torch
from torch._six import container_abcs, string_classes

np_str_obj_array_pattern = re.compile(r"[SaUO]")


def default_convert(data):
    r"""Converts each NumPy array data field into a tensor"""
    elem_type = type(data)
    if isinstance(data, torch.Tensor):
        return data
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        # array of string classes and object
        if (
            elem_type.__name__ == "ndarray"
            and np_str_obj_array_pattern.search(data.dtype.str) is not None
        ):
            return data
        return torch.as_tensor(data)
    elif isinstance(data, container_abcs.Mapping):
        return {key: default_convert(data[key]) for key in data}
    elif isinstance(data, tuple) and hasattr(data, "_fields"):  # namedtuple
        return elem_type(*(default_convert(d) for d in data))
    elif isinstance(data, container_abcs.Sequence) and not isinstance(
        data, string_classes
    ):
        return [default_convert(d) for d in data]
    else:
        return data
