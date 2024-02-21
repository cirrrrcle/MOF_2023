import pandas as pd
from linker1smi_process import *
from config import *
from sklearn.model_selection import train_test_split , RandomizedSearchCV
from sklearn.metrics import mean_squared_error,r2_score,make_scorer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import xgboost as xgb
import pickle

columns_all = [
    # 一些基础的信息，其中mof_num列用来回写到原本的数据，其它的都没用。
    'mof', 'metal', 'linker1smi', 'MOF_num'
    # 这两列是需要预测的，我们先预测temperature列
,'temperature', 'time'
    # 2-1.金属基本信息
    ,'oxidation_state', 'period', 'metal_category' # 这里要用mlb转成向量
    # 2-2.金属情况
    ,'max_metal_energy_orbital', 'max_metal_energy_empty_orbitals'
    , 'ns_metal', 'np_metal', 'n_1p_metal', 'n_1d_metal', 'n_2f_metal'
    , 'ns_metal_empty', 'np_metal_empty', 'n_1p_metal_empty', 'n_1d_metal_empty', 'n_2f_metal_empty'
    # 2-3.离子情况
    , 'max_ion_energy_orbital', 'max_ion_energy_empty_orbitals'
    , 'ns_ions', 'np_ions', 'n_1p_ions', 'n_1d_ions', 'n_2f_ions'
    , 'ns_ions_empty', 'np_ions_empty', 'n_1p_ions_empty', 'n_1d_ions_empty', 'n_2f_ions_empty'
    # 3-1.linker1smi这块
    , 'alkene', 'acetylene', 'primary_amide', 'secondary_amide', 'imine', 'carboxyl', 'carbonyl', 'ether', 'ester', 'peroxide', 'acid_anhydride', 'sulfonic_acid', 'sulfate', 'sulfide', 'sulfone', 'double_bond_sulfur_carbon', 'CSC', 'diphosphate', 'phosphate', 'phosphate_ester', 'phosphonate', 'organo_phosphate', 'organic_phosphorus', 'silicon', 'fluorine', 'halogens'
    , 'benzene', 'pyridine', 'pyrimidine', 'imidazole', 'thiophene', 'furan', 'pyrrole', 'total_aromatic_rings'
    , 'N', 'O', 'S', 'P', 'Si', 'F', 'Cl', 'Br', 'I', 'total_N_O_P_S', 'total_Cl_Br_I', 'total_hetero_atoms'
    # 3-2.分子指纹这块
    , 'morgan_fp', 'rdkit_fp', 'tt_fp', 'tt_fp_dict', 'tt_fp_bit', 'ap_fp', 'avalon_fp']

def r2_score_eval(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)

def r2_score_eval_exp(preds, dtrain):
    labels = dtrain.get_label()
    preds_exp = np.exp(preds)
    labels_exp = np.exp(labels)
    return 'r2', r2_score(labels_exp, preds_exp)

# 这个是预测时间的时候，ln time在随机搜索优化时用的函数
def r2_exp_score(y_true, y_pred):
    y_true_exp = np.exp(y_true)
    y_pred_exp = np.exp(y_pred)
    return r2_score(y_true_exp, y_pred_exp)

# 特征工程
def feature_engineering(df):
    # 先删除最高电子轨道能级及其空轨道数吧，这几个后面的数值都有体现的
    df = df.drop(['max_metal_energy_orbital', 'max_metal_energy_empty_orbitals','max_ion_energy_orbital','max_ion_energy_empty_orbitals'], axis=1)

    # 将每个指纹列拆分为单独的列
    fingerprint_columns = ['morgan_fp', 'rdkit_fp', 'tt_fp', 'tt_fp_dict', 'tt_fp_bit', 'ap_fp', 'avalon_fp']
    # 假设df['rdkit_fp']是存储分子指纹的列
    df['rdkit_fp'] = df['rdkit_fp'].apply(lambda x: [int(bit) for bit in x])

    for col in fingerprint_columns[1:2]:
        # 假设每个指纹是一个位向量，可以使用list表示 
        # 拆分成单独的列
        df_fp = pd.DataFrame(df[col].tolist(), index=df.index)
        df_fp.columns = [f"{col}_{i}" for i in range(df_fp.shape[1])]
        df = pd.concat([df, df_fp], axis=1)

    # 丢弃原始的指纹列
    df.drop(columns=[col for col in fingerprint_columns if col in df.columns], inplace=True)

    # 有些列好像不是合适的类型，需要转换
    cols_to_convert = ['N', 'O', 'S', 'P', 'Si', 'F', 'Cl', 'Br', 'I']
    for col in cols_to_convert:
        if col in df.columns:
            df[col] = df[col].astype(float)
        else:
            df[col] = 0.0
    for col in cols_to_convert:
        df[col] = df[col].astype(float)

    # 金属类别进行向量编码
    mlb = MultiLabelBinarizer()
    # 转换 'metal_category' 列，
    df = df.join(pd.DataFrame(mlb.fit_transform(df.pop('metal_category')),
                              columns=mlb.classes_,
                              index=df.index))
    # print (df.columns.to_list())
    return df

# 后处理温度
def post_treat_temperature_pred(y_pred):
    # 使用numpy的round函数，并乘以5进行四舍五入到最接近的5的倍数
    return np.round(y_pred / 5) * 5

# 后处理时间
def post_treat_time_pred(y_pred):
    # 创建一个新的数组来保存处理后的预测值
    new_y_pred = np.zeros_like(y_pred)

    # 根据上述规则对每个预测值进行处理
    for i, pred in enumerate(y_pred):
        if pred < 1:
            new_y_pred[i] = round(pred / 0.25) * 0.25
        elif pred < 12:
            new_y_pred[i] = round(pred / 6) * 6
        elif pred < 24:
            new_y_pred[i] = round(pred / 4) * 4
        elif pred < 96:
            new_y_pred[i] = round(pred / 12) * 12
        elif pred < 336:
            new_y_pred[i] = round(pred / 24) * 24
        else:
            new_y_pred[i] = round(pred / 168) * 168

    return new_y_pred

# 训练温度
def train_model_temperature(df,randomized_search_cv_n_iter=10000,num_round=1000):
    # 数据预处理，包括主要链接和特征工程（具体实现请参考前面的代码段）
    df = main_linker(df)
    df = feature_engineering(df)

    # 删除不需要的列
    X = df.drop(['mof', 'metal', 'linker1smi', 'MOF_num', 'temperature', 'time'], axis=1)

    # 定义目标变量
    y = df['temperature']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义模型
    model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist', random_state=42)

    # 随机搜索的参数分布
    param_distributions = {
        'max_depth': range(3, 15, 1),
        'learning_rate': np.linspace(0.01, 0.25, 25),
        'subsample': np.linspace(0.4, 0.8, 5),
        'colsample_bytree': np.linspace(0.6, 0.8, 3),
        'gamma': np.linspace(0, 0.1, 6),
        'min_child_weight': np.linspace(1, 6, 11),
        'reg_alpha': np.linspace(0, 0.1, 6),
        'reg_lambda': np.linspace(0, 0.1, 6),
    }

    # 随机搜索，使用R²作为评分标准
    grid_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_jobs=1, n_iter=randomized_search_cv_n_iter, cv=5,
                                     scoring='r2', verbose=2)
    grid_search.fit(X_train, y_train)

    # 输出最佳R²分数
    print('随机搜索最优得分(R²):', grid_search.best_score_)

    # 获取最佳参数
    best_params = grid_search.best_params_
    # 保存为 pickle 文件
    with open(path_temp_params_save, 'wb') as f:
        pickle.dump(best_params, f)

    # with open(path_temp_params_save,'rb') as f:
    #     best_params = pickle.load(f)

    best_params['eta'] = best_params.pop('learning_rate')  # 转换为xgb.train所需格式

    # 使用xgb.train进行训练
    param = best_params
    param['objective'] = 'reg:squarederror'

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    bst = xgb.train(param, dtrain, num_round, evals=[(dtest, 'eval')], feval=r2_score_eval, maximize=True, early_stopping_rounds=100)


    bst.save_model(path_temp_model_save)

    # 读取模型
    # loaded_model = xgb.Booster()
    # loaded_model.load_model(path_temp_model_save)

    # 使用模型进行预测
    y_pred = bst.predict(dtest)
    # 计算R^2值
    r2 = r2_score(y_test, y_pred)

    print(f"R^2 score (Temperature): {r2}")

    y_pred = post_treat_temperature_pred(y_pred)

    # 计算R^2值
    r2_post = r2_score(y_test, y_pred)

    print(f"R^2 score (Temperature) (Post treat): {r2_post}")

    return {
        'randomized_search_cv_n_iter': randomized_search_cv_n_iter,
        'num_round': num_round,
        'R^2 score': r2,
        'R^2 score (Post treat)': r2_post,
        'best_params': best_params
    }


# 训练时间
def train_model_time(df, randomized_search_cv_n_iter=10000, num_round=1000):
    # 数据预处理，包括主要链接和特征工程
    df = main_linker(df)
    df = feature_engineering(df)

    # 温度变换下
    df['temperature'] = 1000 / (df['temperature'] + 273)

    # 删除不需要的列
    X = df.drop(['mof', 'metal', 'linker1smi', 'MOF_num', 'time'], axis=1)

    # 定义目标变量，并对时间取自然对数
    y = np.log(df['time'])

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 定义模型
    model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='gpu_hist', random_state=42)

    # 随机搜索的参数分布
    param_distributions = {
        'max_depth': range(3, 15, 1),
        'learning_rate': np.linspace(0.01, 0.25, 25),
        'subsample': np.linspace(0.4, 0.8, 5),
        'colsample_bytree': np.linspace(0.6, 0.8, 3),
        'gamma': np.linspace(0, 0.1, 6),
        'min_child_weight': np.linspace(1, 6, 11),
        'reg_alpha': np.linspace(0, 0.1, 6),
        'reg_lambda': np.linspace(0, 0.1, 6),
    }

    # 创建一个自定义的评分对象
    custom_r2_scorer = make_scorer(r2_exp_score, greater_is_better=True)

    # 使用自定义评分对象进行随机搜索
    grid_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_jobs=-1,
                                     n_iter=randomized_search_cv_n_iter, cv=5, scoring=custom_r2_scorer)
    # 随机搜索，使用R²作为评分标准
    # grid_search = RandomizedSearchCV(model, param_distributions=param_distributions, n_jobs=-1, n_iter=randomized_search_cv_n_iter, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)

    # 输出最佳R²分数
    print('随机搜索最优得分(R²)，注意这个是ln后的分数而非实际分数:', grid_search.best_score_)

    # 获取最佳参数并保存
    best_params = grid_search.best_params_
    best_params['eta'] = best_params.pop('learning_rate')
    with open(path_time_params_save, 'wb') as f:
        pickle.dump(best_params, f)

    # 使用xgb.train进行训练
    param = best_params
    param['objective'] = 'reg:squarederror'
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    bst = xgb.train(param, dtrain, num_round, evals=[(dtest, 'eval')], feval=r2_score_eval_exp, maximize=True,
                    early_stopping_rounds=100)
    bst.save_model(path_time_model_save)

    # 使用模型进行预测，并还原预测值
    y_pred = bst.predict(dtest)
    y_pred_exp = np.exp(y_pred)
    r2 = r2_score(np.exp(y_test), y_pred_exp)

    print(f"R^2 score (Time): {r2}")

    # 如果有后处理，可以在此处添加
    y_pred_exp = post_treat_time_pred(y_pred_exp)

    # 再次计算R²分数
    r2_post = r2_score(np.exp(y_test), y_pred_exp)
    print(f"R^2 score (Time) (Post treat): {r2_post}")

    return {
        'randomized_search_cv_n_iter': randomized_search_cv_n_iter,
        'num_round': num_round,
        'R^2 score': r2,
        'R^2 score (Post treat)': r2_post,
        'best_params': best_params
    }


def forecast_temperature(path_load_test=path_load_test):
    # 读取模型
    loaded_model_temp = xgb.Booster()
    loaded_model_temp.load_model(latest_temp_model)

    # 使用模型进行预测
    df_test_raw = pd.read_csv(path_load_test)
    # 蠢蠢的加了这么一行，早知道当时不该加的
    df_test_raw['MOF_num'] = df_test_raw.index
    # 数据预处理，包括主要链接和特征工程（具体实现请参考前面的代码段）
    df_test = main_linker(df_test_raw)
    df_test = feature_engineering(df_test)

    # 删除不需要的列
    X = df_test.drop(['mof', 'metal', 'linker1smi', 'MOF_num'], axis=1)

    # 转换为DMatrix
    dtest = xgb.DMatrix(X)

    # 使用温度模型进行预测
    temperature_predictions = loaded_model_temp.predict(dtest)

    # 如果有一个单独的时间模型，使用它进行预测
    # time_predictions = loaded_model_time.predict(dtest)

    # 将预测结果与MOF编号结合
    df_test_raw['temperature'] = temperature_predictions

    # 如果需要对温度进行后处理（例如四舍五入到5℃），可以在此处添加代码
    df_test_raw['temperature'] = post_treat_temperature_pred(df_test_raw['temperature'])

    # 保存为CSV文件

    # prediction_df.to_csv('finger_prediction.csv', index=False)
    return df_test_raw


def forecast_time(df_test_raw):
    # 读取模型
    loaded_model_time = xgb.Booster()
    loaded_model_time.load_model(latest_time_model)

    # 使用模型进行预测
    df_test_raw = df_test_raw.copy()
    df_test = main_linker(df_test_raw)
    df_test = feature_engineering(df_test)

    # 温度变换
    df_test['temperature'] = 1000 / (df_test['temperature'] + 273)

    # 删除不需要的列
    X = df_test.drop(['mof', 'metal', 'linker1smi', 'MOF_num'], axis=1)

    # 转换为DMatrix
    dtest = xgb.DMatrix(X)

    # 使用时间模型进行预测（得到的预测值是对数形式）
    time_predictions_log = loaded_model_time.predict(dtest)
    # 还原预测值
    time_predictions = np.exp(time_predictions_log)

    # 对时间后处理
    time_predictions = post_treat_time_pred(time_predictions)

    # 将预测结果与MOF编号和温度结合
    prediction_df = pd.DataFrame({
        'mof': df_test_raw['mof'],
        'temperature': df_test_raw['temperature'],
        'time': time_predictions,
    })

    # 保存为CSV文件
    prediction_df.to_csv(("../submit/finger_prediction_submit_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), header=None, index=False)

