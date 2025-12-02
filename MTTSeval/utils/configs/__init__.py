from .mimiciv_config import get_mimiciv_config
from .eicu_config import get_eicu_config

def get_config(ehr, **kwargs):
    if ehr == "mimiciv":
        config = get_mimiciv_config()
    elif ehr == "eicu":
        config = get_eicu_config()
    else:
        raise ValueError(f"Unknown EHR dataset: {ehr}")
    # Override with any additional kwargs
    config.update(kwargs)
    return config
