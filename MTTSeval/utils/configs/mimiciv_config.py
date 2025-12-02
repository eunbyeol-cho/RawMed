from .base_config import get_base_config
import numpy as np

def get_mimiciv_config():
    config = get_base_config()
    config.update({
        "ehr": "mimiciv",
        "max_event_size": {6: 165, 12: 243, 24: 366}[config["obs_size"]],
        "table_names": ["labevents", "inputevents", "prescriptions"],
        "predef_vocab": "mimiciv_predef_vocab.pickle",
        "col_type": "mimiciv_col_dtype.pickle",
        "split_file_name": "mimiciv_split.csv",
        "input_file_name": "mimiciv_hi_input.npy",
        "type_file_name": "mimiciv_hi_type.npy",
        "time_file_name": "mimiciv_hi_time.npy",
        "real_time_file_name": "mimiciv_hi_num_time.npy",
        "task_string": {
            "creatinine": ["creatinine"],
            "platelets": ["platelet count"],
            "wbc": ["wbc count", "white blood cells"],
            "hb": ["hemoglobin"],
            "bicarbonate": ["bicarbonate"],
            "sodium": ["sodium"],
        },
        "task_config": {
            "lab": {
                "table_name": "labevents",
                "itemid_col": "itemid",
                "value_col": "valuenum",
            },
            "input": {
                "table_name": "inputevents",
                "itemid_col": "itemid",
                "task": ["norepinephrine", "propofol"],
            },
            "med": {
                "table_name": "prescriptions",
                "itemid_col": "drug",
                "task": ["magnesium sulfate", "heparin", "potassium chloride|kcl"],
            },
        },
        "lab_test_bins": {
            "creatinine": [-np.inf, 1.2, 2.0, 3.5, 5, np.inf],
            "platelets": [-np.inf, 20, 50, 100, 150, np.inf],
            "wbc": [-np.inf, 4, 12, np.inf],
            "hb": [-np.inf, 8, 10, 12, np.inf],
            "bicarbonate": [-np.inf, 22, 29, np.inf],
            "sodium": [-np.inf, 135, 145, np.inf],
        },
        "lab_labels": {
            "creatinine": [0, 1, 2, 3, 4],
            "platelets": [4, 3, 2, 1, 0],
            "wbc": [0, 1, 2],
            "hb": [0, 1, 2, 3],
            "bicarbonate": [0, 1, 2],
            "sodium": [0, 1, 2],
        },
        "pred_tasks": [
            'creatinine', 'platelets', 'wbc', 'hb', 'bicarbonate', 'sodium',
            'magnesium sulfate', 'heparin', 'potassium chloride|kcl',
            'norepinephrine', 'propofol'
        ],
        "pid_column": "stay_id",
        "time_column": "time",
        "split_column": "seed0",
        "item_column": {
            "labevents": "itemid",
            "inputevents": "itemid",
            "prescriptions": "drug",
        }
    })
    return config 