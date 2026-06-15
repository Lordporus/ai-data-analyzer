import json
import base64
import pandas as pd
from dataclasses import is_dataclass, asdict

class PipelineEncoder(json.JSONEncoder):
    def default(self, obj):
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, pd.DataFrame):
            import io
            buffer = io.BytesIO()
            obj.to_parquet(buffer)
            return {"_type": "dataframe", "data": base64.b64encode(buffer.getvalue()).decode('utf-8')}
        # If it has a summary_dict method, it's the root PipelineResult but not a dataclass?
        if hasattr(obj, "summary_dict"):
            # Attempt to convert to dict manually if it's not recognized as a dataclass
            return obj.__dict__
        return super().default(obj)

def pipeline_decoder(dct):
    if "_type" in dct and dct["_type"] == "dataframe":
        import io
        return pd.read_parquet(io.BytesIO(base64.b64decode(dct["data"])))
    return dct

class Dict2Obj:
    """Recursively wraps a dictionary so that its keys can be accessed as attributes."""
    def __init__(self, dictionary):
        if not isinstance(dictionary, dict):
            return
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, Dict2Obj(value))
            elif isinstance(value, list):
                setattr(self, key, [Dict2Obj(x) if isinstance(x, dict) else x for x in value])
            else:
                setattr(self, key, value)
    
    def __getattr__(self, name):
        # Return None for missing attributes instead of AttributeError to mimic optional dataclass fields
        return None

def dump_pipeline_result(result, path):
    with open(path, "w") as f:
        json.dump(result, f, cls=PipelineEncoder)

def load_pipeline_result(path):
    with open(path, "r") as f:
        data = json.load(f, object_hook=pipeline_decoder)
    return Dict2Obj(data)

def loads_pipeline_result(content_bytes):
    data = json.loads(content_bytes.decode('utf-8'), object_hook=pipeline_decoder)
    return Dict2Obj(data)
