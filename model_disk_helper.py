import re
import warnings
import os

from torch import save, load
from torch.nn import Module
from datetime import datetime
from model import CustomModel
from typing import List, Dict, Tuple

_base_path = "./results/"
_delim = "_"
_auto_legacy_check_default = False
_auto_legacy_check_reference = datetime.strptime("2024-06-14-18:10:00", '%Y-%m-%d-%H:%M:%S')

def save_model(model: Module, numEpochs: int = 0, trainingTime: float = 0, legacy: bool = False):
    if legacy:
        return save_legacy(model)
    
    filename = model.name if model.name else ""
    path = _get_path(filename)
    disk_object = {
        'model_state': model.state_dict(),
        'metadata': _construct_metadata(model, numEpochs, trainingTime)
    }
    save(disk_object, path)
    return path

    
def load_model(filename: str, legacy: bool = False, legacy_auto_check: bool = True) -> Tuple[Module, dict]:
    legacy_auto_check_flag = legacy_auto_check and _check_for_legacy(filename)
    if legacy or legacy_auto_check_flag:
        return load_legacy(filename), None
    
    path = _reconstruct_path(filename)
    disk_object = load(path)
    return disk_object['model_state'], disk_object['metadata']

def save_legacy(model: Module) -> str:
    path = _get_path()
    save(model.state_dict(), path)
    return path

def load_legacy(model_name: str):
    path = _reconstruct_path(model_name)
    return load(path)

def enhance_metadata_with_metric(filename: str, hit, mrr):    
    path = _reconstruct_path(filename)
    disk_object = load(path)
    metric = {
        'hit': hit,
        'mrr': mrr
    }
    disk_object['metadata'] = disk_object['metadata'] | metric

    save(disk_object, path)

def overwrite_base_path(base_path: str):
    _base_path = base_path

def _construct_metadata(model: Module, numEpochs: int = 0, trainingTime: float = 0,) -> dict:
    metadata = {}
    # if isinstance(model, CustomModel):
    #     metadata['device'] = model.device
    metadata['device'] = model.device
    #metadata['hardwareName'] = "NVIDIA GeForce RTX 2060"
    metadata['name'] = model.name
    metadata['numEpochs'] = numEpochs
    metadata['trainingTime'] = trainingTime
    metadata['savedAt'] = _get_stamp()

    return metadata

def _get_stamp() -> str:
    return datetime.today().strftime('%Y-%m-%d-%H:%M:%S')

def _get_path(filename: str = "") -> str:
    if filename != "":
        return _base_path + filename + _delim + _get_stamp()
    return _base_path + _get_stamp()

def _reconstruct_path(name: str) -> str:
    return _base_path + name

def _check_for_legacy(filename: str) -> bool:
    timestamp_str = _extract_timestamp(filename)
    
    if timestamp_str == "":
        return _auto_legacy_check_default
    
    timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d-%H:%M:%S')

    return True if timestamp < _auto_legacy_check_reference else False

def _extract_timestamp(filename: str) -> str:
    pattern = r'(\d{4}-\d{2}-\d{2}-\d{2}:\d{2}:\d{2})'

    matches = re.findall(pattern, filename)

    if matches:
        timestamp_str = matches[len(matches)-1]
        return timestamp_str
    else:
        message = "No timestamp found in {fn}. Auto legacy check failed. Assume {lt}"
        lt = "legacy" if _auto_legacy_check_default else "no legacy"
        warnings.warn(message.format(fn=filename,lt=lt))
        return ""