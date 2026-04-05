# import sys
# import json
# import nn_fit as fit

# params_json = sys.argv[1]
# p = json.loads(params_json)

# fit.main(p)

import sys
import json
import traceback
import nn_fit as fit

def restore_tuples(obj):
    if isinstance(obj, list):
        return tuple(restore_tuples(i) for i in obj)
    elif isinstance(obj, dict):
        return {k: restore_tuples(v) for k, v in obj.items()}
    else:
        return obj
    
try:
    params_json = sys.argv[1]
    p = json.loads(params_json)
    p = restore_tuples(p)
    fit.main(p)

except Exception:
    print("\n FIT CRASHED \n", flush=True)
    traceback.print_exc()
    sys.exit(1)