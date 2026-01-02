from runpod.serverless.utils.rp_validator import validate

def _one_of(allowed):
    return lambda x: x in allowed

def _non_empty_str(x):
    return isinstance(x, str) and len(x.strip()) > 0

def validate_input(job_input: dict):
    schema = {
        "request_id": {"type": str, "required": False, "default": None},

        "image": {
            "type": dict,
            "required": True,
        },

        "target": {
            "type": str,
            "required": True,
            "constraints": lambda s: _non_empty_str(s) and len(s) <= 64,
        },

        "selection": {"type": dict, "required": False, "default": {}},

        "output": {"type": dict, "required": False, "default": {}},

        "postprocess": {"type": dict, "required": False, "default": {}},
    }

    res = validate(job_input, schema)
    return res

