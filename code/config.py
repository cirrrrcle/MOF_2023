import pandas as pd
from correct_elec_cfg import ElectronConfigurations
import os
import time
import glob
import datetime

'''
1.各种配置参数设置
2.必须的清洗工作
'''

def get_latest_file(path_save_main, pattern):
    # 使用glob列出所有匹配的文件
    files = glob.glob(os.path.join(path_save_main, pattern))

    # 如果没有找到匹配的文件，返回None
    if not files:
        return None

    # 从文件名中提取时间戳并找到最新的文件
    latest_file = max(files, key=lambda x: os.path.basename(x).split('_')[1])
    return latest_file


path_load_main = r'../data/DatasetA'
path_load_train = os.path.join(path_load_main,'train/finger_train.csv')
path_load_test = os.path.join(path_load_main,'test/finger_test.csv')
path_load_assist = os.path.join('../data/External/electronic_configuration.csv')

path_save_main = r'../submit/'
if not os.path.exists(path_save_main):
    os.makedirs(path_save_main)

ts = time.strftime('%Y-%m-%d %H%M%S')
path_temp_model_save = path_save_main + '/temperature_{}.model'.format(ts)
path_temp_params_save = path_save_main + '/temperature_{}.pkl'.format(ts)
path_time_model_save = path_save_main + '/time_{}.model'.format(ts)
path_time_params_save = path_save_main + '/time_{}.pkl'.format(ts)
path_save_finger_prediction = path_save_main + '/finger_prediction_{}.csv'.format(ts)

# 获取最新的temperature的model文件
latest_temp_model = get_latest_file(path_save_main, 'temperature_*.model')
# print(latest_temp_model)

# 获取最新的temperature的pkl文件
latest_temp_params = get_latest_file(path_save_main, 'temperature_*.pkl')
# print(latest_temp_params)

# 获取最新的time的model文件
latest_time_model = get_latest_file(path_save_main, 'time_*.model')
# print(latest_time_model)

# 获取最新的time的pkl文件
latest_time_params = get_latest_file(path_save_main, 'time_*.pkl')
# print(latest_time_params)


# Define the columns you are interested in
cols = ['mof', 'metal', 'linker1smi', 'oxidation_state', 'temperature', 'time']

df = pd.read_csv(path_load_train)
# Select only the columns that exist in the data frame
df = df[[col for col in cols if col in df.columns]]

df_test = pd.read_csv(path_load_test)
df_test = df_test[[col for col in cols if col in df_test.columns]]
# df = df.append(df_test)
df = df.reset_index(drop=True)

electron_config_columns = ['1s', '2s', '3s', '4s', '5s', '6s', '7s',
                                  '2p', '3p', '4p', '5p', '6p',
                                  '3d', '4d', '5d', '6d',
                                  '4f', '5f']

s_orbital_cols = ['1s', '2s', '3s', '4s', '5s', '6s', '7s']

# 原始的分类字典
metal_categories = {
    # 碱金属和碱土金属，外层电子都是容易失去的，而且都很像，In其实不应该被放进来，但是失去电子后和碱金属很像，就这么样吧
    'I-IIA': ['H', 'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr', 'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra'
        , 'In'],
    # IB和IIB族
    'I-IIB': ['Cu', 'Ag', 'Au', 'Rg', 'Zn', 'Cd', 'Hg', 'Cn'
        , 'Mn'],
    # VIII族,Mn其实也不应该放进来，但是其氧化态来看，既在I-IIB，又在VIII
    'VIII': ['Fe', 'Co', 'Ni', 'Rh', 'Ru', 'Pd', 'Os', 'Ir', 'Pt', 'Hs', 'Mt', 'Ds'
        , 'Mn'],
    # 稀土系列金属 Zr和In其实不应该被放进来，但和Th一样都是四价，暂且如此吧，理解为第四周期的稀土元素吧
    'La': ['La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu'
        , 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr'
        , 'Y'
        , 'Zr', 'V', 'Ti']
    }



columns_all = ['mof', 'metal', 'linker1smi', 'oxidation_state', 'temperature', 'time', 'MOF_num', 'period', 'metal_category', 'max_metal_energy_orbital', 'max_metal_energy_empty_orbitals', 'ns_metal', 'np_metal', 'n_1p_metal', 'n_1d_metal', 'n_2f_metal', 'ns_metal_empty', 'np_metal_empty', 'n_1p_metal_empty', 'n_1d_metal_empty', 'n_2f_metal_empty', 'max_ion_energy_orbital', 'max_ion_energy_empty_orbitals', 'ns_ions', 'np_ions', 'n_1p_ions', 'n_1d_ions', 'n_2f_ions', 'ns_ions_empty', 'np_ions_empty', 'n_1p_ions_empty', 'n_1d_ions_empty', 'n_2f_ions_empty', 'alkene', 'acetylene', 'primary_amide', 'secondary_amide', 'imine', 'carboxyl', 'carbonyl', 'ether', 'ester', 'peroxide', 'acid_anhydride', 'sulfonic_acid', 'sulfate', 'sulfide', 'sulfone', 'double_bond_sulfur_carbon', 'CSC', 'diphosphate', 'phosphate', 'phosphate_ester', 'phosphonate', 'organo_phosphate', 'organic_phosphorus', 'silicon', 'fluorine', 'halogens', 'benzene', 'pyridine', 'pyrimidine', 'imidazole', 'thiophene', 'furan', 'pyrrole', 'total_aromatic_rings', 'N', 'O', 'S', 'P', 'Si', 'F', 'Cl', 'Br', 'I', 'total_N_O_P_S', 'total_Cl_Br_I', 'total_hetero_atoms']

# df显示设置
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


def get_correct_elec_cfg(metal,filename='../data/External/electronic_configuration.csv'):
    """
    Returns the correct electron configuration for the given configuration.
    :param electron_configuration: The given electron configuration.
    :return: The correct electron configuration.
    """
    configs = ElectronConfigurations(filename)
    return configs.query(metal)

def clean_data(df):
    # MOF名称有重复行，用新列来表征下
    df['MOF_num'] = df.index
    '''
    # 部分电子数排布是错的，尤其是6p上面数据大于6的全部都是把3d的电子排到6p上面去了
    # Correct the data in '6p' and '3d' columns
    # df.loc[df['6p'] >= 7, ['6p', '3d']] = df.loc[df['6p'] >= 7, ['3d', '6p']].values
    # df.to_excel('1.xlsx')
    # print (df[['metal','6p','3d']])
    '''
    # Apply the function to the 'metal' column and create new columns
    electron_config_df = df['metal'].apply(get_correct_elec_cfg).apply(pd.Series)

    # Rename the columns
    electron_config_df.columns = electron_config_columns

    # Join the new columns to the original dataframe
    df_extended = pd.concat([df, electron_config_df], axis=1)

    return df_extended

df = clean_data(df)