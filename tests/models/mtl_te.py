# import pytest
import tensorflow as tf

from deepctr.models.multitask import SharedBottom, ESMM, MMOE, PLE
from utils_mtl import get_mtl_test_data, check_mtl_model
def test_ESMM():
    if tf.__version__ == "1.15.0":  # slow in tf 1.15
        return
    model_name = "ESMM"
    x, y_list, dnn_feature_columns = get_mtl_test_data()
    print("x",x)
    print("y_list",y_list)
    model = ESMM(dnn_feature_columns, tower_dnn_hidden_units=(8,), task_types=['binary', 'binary'],
                 task_names=['label_marital', 'label_income'])
    check_mtl_model(model, model_name, x, y_list, task_types=['binary', 'binary'])
test_ESMM()