"""
Helper functions from Olive's other project (gene-circuit-glitch-prediction)
"""

import os
import json
import numpy as np


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
                flat_list.extend(l)
            else:
                flat_list.append(l)
        return flat_list
    else:
        return [item for sublist in listlike for item in sublist]


def get_unique_flat(listlike):
    u = list(set(flatten_listlike(listlike)))
    u.sort()
    return u

def per_mol_to_per_molecules(per_mol):
    """ Translate a value from the unit of per moles to per molecules.
    The number of M of mRNA in a cell was calculated using the average 
    number of mRNA in an E. coli cell (100 molecules) and the average volume of an E.
    coli cell (1.1e-15 L) to give ca. 1 molecule ~ 1.50958097 nM ~ 1.50958097e-9 M"""
    # 1/mol to 1/molecule
    # return np.divide(jmol, SCIENTIFIC['mole'])
    return np.multiply(per_mol, 1.50958097/np.power(10, 9))
