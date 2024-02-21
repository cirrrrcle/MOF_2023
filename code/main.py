###This is for finger
import logging
import time
from model_utils import *

if not os.path.exists("./logs"):
    os.makedirs("./logs")

# # Create a logger
# logger = logging.getLogger('main_logger')
# logger.setLevel(logging.INFO)  # Log all escalated at and above this level
# # Create a file handler
# handler = logging.FileHandler('./logs/main.log')
# handler.setLevel(logging.INFO)
# # Create a logging format
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# # Add the handlers to the logger
# logger.addHandler(handler)


def main_model(df):
    start_time = time.time()
    train_model_temperature(df,randomized_search_cv_n_iter=30000,num_round=30000)
    train_model_time(df,randomized_search_cv_n_iter=30000,num_round=30000)

    # logger.info(f"Temperature model results: {temp_results}")
    # logger.info(f"Time model results: {time_results}")

    df_test_raw = forecast_temperature(path_load_test)
    # 预测时间的时候需要输入预测的温度作为参数。预测好直接直接保存finger_prediction.csv了。
    forecast_time(df_test_raw)

    # end_time = time.time()
    # total_time = end_time - start_time
    # logger.info(f"Total running time: {total_time:.2f} seconds")

###This is for RAC
import datetime, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV

columns_pd = ['temperature', 'time', 'param1', 'param2', 'param3', 'param4', 'param5', 'additive']
columns_pd_1 = ['temperature', 'time', 'param1', 'param2', 'param3', 'param4', 'param5', 'additive_category']
patterns = [r'temp_.*\.py', r'time_.*\.py',
            r'param1_.*\.py', r'param2_.*\.py', r'param3_.*\.py', r'param4_.*\.py', r'param5_.*\.py',
            r'additive_.*\.py']

def load_data():
    rac_test = pd.read_csv('../data/DatasetB/test/RAC_test.csv')
    rac_train = pd.read_csv('../data/DatasetB/train/RAC_train.csv')
    feature_names = [col for col in rac_test.columns if col != 'mof']

    features = rac_train[feature_names]
    features_test = rac_test[feature_names]

    targets = rac_train[['temperature', 'time', 'param1', 'param2', 'param3', 'param4', 'param5', 'additive_category']]

    features = features.to_numpy().astype(np.float64)
    targets = targets.to_numpy().astype(np.float64)
    features_test = features_test.to_numpy().astype(np.float64)

    return features, targets, features_test, feature_names

features, targets, features_test, feature_names = load_data()

results = []

from sklearn.svm import LinearSVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import Binarizer
import pandas as pd
from sklearn.pipeline import make_union
from tpot.builtins import ZeroCount
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from tpot.export_utils import set_param_recursive

def post_treat_temperature_pred(y_pred):
    # 创建一个新的数组来保存处理后的预测值
    new_y_pred = np.zeros_like(y_pred)
    # 根据上述规则对每个预测值进行处理
    for i, pred in enumerate(y_pred):
        if pred < 5:
            new_y_pred[i] = 5
        elif pred < 180:
            new_y_pred[i] = round(pred / 5) * 5
        else:
            new_y_pred[i] = round(pred / 20) * 20
    return new_y_pred

#temperature
print('Fitting temperature...')
exported_pipeline = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    XGBRegressor(learning_rate=0.1, max_depth=6, min_child_weight=9, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.4, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 7)


exported_pipeline.fit(features, targets[:, 0])
result = exported_pipeline.predict(features_test)
result = post_treat_temperature_pred(result)
results.append(result)

#time
print('Fitting time...')
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.001, loss="linear", n_estimators=100)),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=10, min_samples_leaf=15, min_samples_split=13)),
    DecisionTreeRegressor(max_depth=1, min_samples_leaf=15, min_samples_split=17)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 5)


exported_pipeline.fit(features, targets[:, 1])
result = exported_pipeline.predict(features_test)
results.append(result)

#param1
print('Fitting param1...')
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=2, min_child_weight=15, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.6500000000000001, verbosity=0)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.15000000000000002, min_samples_leaf=1, min_samples_split=4, n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 7)


exported_pipeline.fit(features, targets[:, 2])
result = exported_pipeline.predict(features_test)
results.append(result)

#param2
print('Fitting param2...')
exported_pipeline = make_pipeline(
    ZeroCount(),
    MaxAbsScaler(),
    XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=10, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.35000000000000003, verbosity=0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 7)


exported_pipeline.fit(features, targets[:, 3])
result = exported_pipeline.predict(features_test)
results.append(result)

#param3
print('Fitting param3...')
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVR(C=1.0, dual=False, epsilon=0.01, loss="squared_epsilon_insensitive", tol=0.01)),
    Binarizer(threshold=0.2),
    AdaBoostRegressor(learning_rate=0.001, loss="square", n_estimators=100)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)


exported_pipeline.fit(features, targets[:, 4])
result = exported_pipeline.predict(features_test)
results.append(result)


#param4
print('Fitting param4...')
exported_pipeline = GradientBoostingRegressor(alpha=0.75, learning_rate=0.1, loss="huber", max_depth=8, max_features=0.9500000000000001, min_samples_leaf=12, min_samples_split=7, n_estimators=100, subsample=0.3)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 7)


exported_pipeline.fit(features, targets[:, 5])
result = exported_pipeline.predict(features_test)
results.append(result)

#param5
print('Fitting param5...')
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=1, min_samples_leaf=7, min_samples_split=2)),
    MinMaxScaler(),
    LinearSVR(C=0.5, dual=False, epsilon=0.001, loss="squared_epsilon_insensitive", tol=0.001)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 7)


exported_pipeline.fit(features, targets[:, 6])
result = exported_pipeline.predict(features_test)
results.append(result)

#additive
print('Fitting additive...')
exported_pipeline = XGBClassifier(learning_rate=0.5, max_depth=8, min_child_weight=13, n_estimators=100, n_jobs=1, subsample=0.45, verbosity=0)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)


exported_pipeline.fit(features, targets[:, 7])
result = exported_pipeline.predict(features_test)
results.append(result)

def save_results(results):
    df = pd.DataFrame()

    df['mof'] = [i for i in range(1, len(results[0]) + 1)]
    for i,col in enumerate(columns_pd):
        df[col] = results[i]

    df.to_csv(("../submit/RAC_prediction_submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)

save_results(results)
