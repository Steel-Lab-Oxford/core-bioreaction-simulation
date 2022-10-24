"""
Helper functions from Olive's other project (gene-circuit-glitch-prediction)
"""

import os
import json


def process_json(json_dict):
    for k, v in json_dict.items():
        if v == "None":
            json_dict[k] = None
    return json_dict


def load_json_as_dict(json_pathname: str, process=True) -> dict:
    if not json_pathname:
        return {}
    elif type(json_pathname) == dict:
        jdict = json_pathname
    elif type(json_pathname) == str:
        if os.stat(json_pathname).st_size == 0:
            jdict = {}
        else:
            file = open(json_pathname)
            jdict = json.load(file)
            file.close()
    else:
        raise TypeError(
            f'Unknown json loading input {json_pathname} of type {type(json_pathname)}.')
    if process:
        return process_json(jdict)
    return jdict


def flatten_listlike(listlike, safe=False):
    if safe:
        flat_list = []
        for l in listlike:
            if type(l) == tuple or type(l) == list:
                flat_list.extend(*l)
            else:
                flat_list.append(l)
    else:
        return [item for sublist in listlike for item in sublist]


def get_unique_flat(listlike):
    u = list(set(flatten_listlike(listlike)))
    u.sort()
    return u
