import json
import gvar as gv

def serialize_key(key):
    return json.dumps(key)

def serialize_gvar(val):
    return {"mean": float(gv.mean(val)), "sdev": float(gv.sdev(val))}

def save_json_priors(priors, filename="nn_priors.json"):
    data = {}
    for key, val in priors.items():
        skey = serialize_key(key)
        sval = serialize_gvar(val)
        data[skey] = sval

    with open(filename, "w") as f:
        json.dump(data, f, indent=2)

def list_to_tuple(obj):
    if isinstance(obj, list):
        return tuple(list_to_tuple(x) for x in obj)
    return obj

def deserialize_key(skey):
    return list_to_tuple(json.loads(skey))

def deserialize_gvar(d):
    return gv.gvar(d["mean"], d["sdev"])

def load_json_priors(filename="nn_priors.json"):
    with open(filename, "r") as f:
        data = json.load(f)

    priors = {}
    for skey, sval in data.items():
        key = deserialize_key(skey)
        val = deserialize_gvar(sval)
        priors[key] = val

    return priors
