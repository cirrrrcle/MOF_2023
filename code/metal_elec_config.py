from config import *

# 0-1. 获取外层电子组态，包括ns、np、n-1p、n-1d、n-2f
# Function to get the electron configuration for each period
'''
这段思想比较复杂，我在这里详细列举下
首先找到最大的有电子的ns层，如6s上有1个电子，则n=6,然后计算其ns np (n-1)d (n-2)f 轨道上的电子数，作为其外层电子数
2023.7.31更新，我直接列出ns np n-1p n-1d n-2f 5个轨道作为外层电子。其中
注意如果没有d轨道或者f轨道，则相应的数值填写-1，以示区分。
'''
def get_electron_config(row):
    # Define the orbitals for each period
    orbitals = {1: ['1s'], 2: ['2s', '2p'], 3: ['3s', '3p', '3d'], 4: ['4s', '4p', '4d', '4f'],
                5: ['5s', '5p', '5d', '5f'], 6: ['6s', '6p', '6d'], 7: ['7s', '7p']}

    period = row['period']
    ns_orbital = orbitals[period][0] if period in orbitals else None
    np_orbital = orbitals[period][1] if period in orbitals and len(orbitals[period]) > 1 else None
    n_1p_orbital = orbitals[period - 1][1] if period - 1 in orbitals and len(orbitals[period - 1]) > 1 else None
    n_1d_orbital = orbitals[period - 1][2] if period >= 4 and period - 1 in orbitals and len(
        orbitals[period - 1]) > 2 else None
    n_2f_orbital = orbitals[period - 2][3] if period >= 5 and period - 2 in orbitals and len(
        orbitals[period - 2]) > 3 else None

    ns_electrons = row[ns_orbital] if ns_orbital and ns_orbital in row else -1
    np_electrons = row[np_orbital] if np_orbital and np_orbital in row else -1
    n_1p_electrons = row[n_1p_orbital] if n_1p_orbital and n_1p_orbital in row else -1
    n_1d_electrons = row[n_1d_orbital] if n_1d_orbital and n_1d_orbital in row else -1
    n_2f_electrons = row[n_2f_orbital] if n_2f_orbital and n_2f_orbital in row else -1

    return pd.Series([ns_electrons, np_electrons, n_1p_electrons, n_1d_electrons, n_2f_electrons])

# 0-2.根据电子失去的规则，返回到金属离子对应的电子分布
def calculate_metal_ion_configuration(row):
    # Define the order of orbitals for removing electrons
    order_of_orbitals = ['7s', '6d', '6p', '6s', '5f', '5d', '5p', '5s', '4f', '4d', '4p', '4s', '3d', '3p', '3s',
                         '2p', '2s', '1s']

    # Get the oxidation state of the metal ion
    oxidation_state = row['oxidation_state']

    # Copy the row data to a new variable so as not to modify the original data
    row_data = row.copy()

    # Iterate over the orbitals in the specified order
    for orbital in order_of_orbitals:
        # If the orbital is not in the row data (i.e., the metal does not have this orbital), skip it
        if orbital not in row_data:
            continue

        # Get the number of electrons in the current orbital
        num_electrons = row_data[orbital]

        # If the number of electrons in the current orbital is less than or equal to the oxidation state
        if num_electrons <= oxidation_state:
            # Subtract the number of electrons in the current orbital from the oxidation state
            oxidation_state -= num_electrons
            # Set the number of electrons in the current orbital to 0
            row_data[orbital] = 0
        else:
            # If the number of electrons in the current orbital is greater than the oxidation state
            # Subtract the oxidation state from the number of electrons in the current orbital
            row_data[orbital] -= oxidation_state
            # Set the oxidation state to 0
            oxidation_state = 0

        # If the oxidation state has reached 0, break the loop
        if oxidation_state == 0:
            break

    # Return the updated row data
    return row_data

# 0-3.根据最高能量的轨道名称，计算其最高能量的空轨道数
def calculate_max_energy_empty_orbitals(row):
    max_energy_orbital = row['max_energy_orbital']
    orbital_type = max_energy_orbital[-1]  # Get the type of the orbital (s, p, d, or f)
    num_electrons = row[max_energy_orbital]  # Get the number of electrons in the orbital
    max_electrons_in_orbital = {'s': 2, 'p': 6, 'd': 10, 'f': 14}
    num_empty_orbitals = (max_electrons_in_orbital[orbital_type] - num_electrons) / 2
    return num_empty_orbitals

# 1-1.确定金属所属周期
def metal_period(df):
    # Determine the period of the element
    df['period'] = df[s_orbital_cols].apply(
        lambda row: max((int(col[0]), val) for col, val in row.items() if val > 0)[0], axis=1)
    return df

# 1-2.确定金属所属类型，稀土金属、碱金属/碱土金属、IB-IIB贵金属、VIII金属，还有三个Mn、In、V、Ti不好处理
def metal_type(df):
    # 将字典改为每个金属对应一个分类列表
    new_metal_categories = dict()
    for category, metals in metal_categories.items():
        for metal in metals:
            if metal not in new_metal_categories:
                new_metal_categories[metal] = [category]
            else:
                new_metal_categories[metal].append(category)

    # 函数改为返回分类列表
    df['metal_category'] = df['metal'].apply(lambda metal: new_metal_categories.get(metal, ['Other']))
    # 测试下锰这一行
    # df = df[df['metal'] == 'Mn']
    # print (df[['metal', 'metal_category']])
    # 测试下Mn的分类
    return df

# 1-3.计算金属能量最大的轨道
def metal_max_energy_orbital(df):
    energy_levels_adjusted = {'1s': 1, '2s': 2, '3s': 3, '4s': 4, '5s': 5, '6s': 6,
                              '2p': 3, '3p': 4, '4p': 5, '5p': 6, '6p': 7,
                              '3d': 5, '4d': 6, '5d': 7, '6d': 8,
                              '4f': 7, '5f': 8
                              }
    df['max_energy_orbital'] = df[energy_levels_adjusted.keys()].apply(
        lambda row: max((energy_levels_adjusted[col], col, val) for col, val in row.items() if val > 0)[1], axis=1)
    return df

# 废弃：找到能量最高轨道上的空轨道数,这一块已经废弃不用了
def abandon_metal_max_energy_empty_orbitals(df):
    max_electrons_in_orbital = {'s': 2, 'p': 6, 'd': 10, 'f': 14}

    # Calculate the number of empty orbitals in the maximum energy level for each metal
    def max_energy_empty_orbitals(row):
        orbital_type = row['max_energy_orbital'][-1]  # Get the type of the orbital (s, p, d, or f)
        num_electrons = row[row['max_energy_orbital']]  # Get the number of electrons in the orbital
        # 思前想后，这里还是应该要保留单电子的，以示区别
        num_empty_orbitals = (max_electrons_in_orbital[orbital_type] - num_electrons) / 2
        return num_empty_orbitals

    df['max_energy_empty_orbitals'] = df.apply(max_energy_empty_orbitals, axis=1)
    return df

# 废弃：找到金属元素的电子组态，如Zn可以表示为[Ar]3d104s2，这个不太适合第六周期及以上的元素
def abandon_electron_configuration_to_symbol(config):
    # Define the noble gas configurations
    noble_gas_configs = {'He': ['1s2'], 'Ne': ['1s2', '2s2', '2p6'], 'Ar': ['1s2', '2s2', '2p6', '3s2', '3p6']}
    # Define the electron filling order
    filling_order = ['1s', '2s', '2p', '3s', '3p', '4s', '3d', '4p', '5s', '4d', '5p', '6s', '4f', '5d', '6p', '7s', '5f', '6d', '7p']
    # Initialize the symbol configuration
    symbol_config = ''
    # Initialize the total number of electrons
    total_electrons = 0
    for orbital in filling_order:
        # Get the number of electrons in this orbital
        num_electrons = config.get(orbital, 0)
        if num_electrons > 0:
            # Add the electrons to the total
            total_electrons += num_electrons
            # Check if we can replace the configuration with a noble gas symbol
            for gas, gas_config in noble_gas_configs.items():
                if all(orbital+str(config.get(orbital, 0)) in gas_config for orbital in filling_order if config.get(orbital, 0) > 0):
                    # Replace the configuration with the noble gas symbol
                    symbol_config = '[' + gas + ']'
                    # Remove the noble gas configuration from the current configuration
                    for gas_orbital in gas_config:
                        config[gas_orbital[0:2]] -= int(gas_orbital[2:])
            # Add the current orbital to the symbol configuration
            symbol_config += orbital + str(num_electrons)
    return symbol_config

# 1-4.计算每个金属的外层电子排布
def metal_outer_elec_config(df):
    # Apply the function to each row and add new columns
    df[['ns_metal', 'np_metal', 'n_1p_metal', 'n_1d_metal','n_2f_metal']] = df.apply(get_electron_config, axis=1)
    return df

# 1-5.计算每个金属的外层电子空轨道数
def metal_outer_elec_empty_orbitals(df):
    # Calculate the number of empty orbitals
    df['ns_metal_empty'] = df['ns_metal'].apply(lambda x: (2 - x) / 2 if x != -1 else -1)
    df['np_metal_empty'] = df['np_metal'].apply(lambda x: (6 - x) / 2 if x != -1 else -1)
    df['n_1p_metal_empty'] = df['n_1p_metal'].apply(lambda x: (6 - x) / 2 if x != -1 else -1)
    df['n_1d_metal_empty'] = df['n_1d_metal'].apply(lambda x: (10 - x) / 2 if x != -1 else -1)
    df['n_2f_metal_empty'] = df['n_2f_metal'].apply(lambda x: (14 - x) / 2 if x != -1 else -1)
    return df

# 1-6.计算各金属氧化态的离子排布
def metal_ions_elec_config(df):
    # Apply the function to calculate the electronic configuration of metal ions
    df_metal_ion = df.apply(lambda row: calculate_metal_ion_configuration(row), axis=1)

    return df_metal_ion[electron_config_columns + ['MOF_num', 'period']]

# 1-7.计算金属离子的外层电子排布
def metal_ions_outer_elec_config(df_metal_ion):
    # Apply the function to each row and add new columns
    df_metal_ion[['ns_ions', 'np_ions', 'n_1p_ions', 'n_1d_ions', 'n_2f_ions']] = df_metal_ion.apply(get_electron_config, axis=1)
    # allow_list = electron_config_columns + ['MOF_num', 'period', 'ns_ions', 'np_ions', 'n_1p_ions', 'n_1d_ions', 'n_2f_ions']
    # print(df_metal_ion[allow_list])
    return df_metal_ion

# 1-8.计算金属离子的外层电子空轨道数
def metal_ions_outer_elec_empty_orbitals(df_metal_ion):
    # Calculate the number of empty orbitals
    df_metal_ion['ns_ions_empty'] = df_metal_ion['ns_ions'].apply(lambda x: (2 - x) / 2 if x != -1 else -1)
    df_metal_ion['np_ions_empty'] = df_metal_ion['np_ions'].apply(lambda x: (6 - x) / 2 if x != -1 else -1)
    df_metal_ion['n_1p_ions_empty'] = df_metal_ion['n_1p_ions'].apply(lambda x: (6 - x) / 2 if x != -1 else -1)
    df_metal_ion['n_1d_ions_empty'] = df_metal_ion['n_1d_ions'].apply(lambda x: (10 - x) / 2 if x != -1 else -1)
    df_metal_ion['n_2f_ions_empty'] = df_metal_ion['n_2f_ions'].apply(lambda x: (14 - x) / 2 if x != -1 else -1)
    # print (df_metal_ion)
    return df_metal_ion

def main_metal(df):
    # 1-1.公用 确定金属所属周期
    df = metal_period(df)
    # 1-2.公用 确定金属所属类型，稀土金属、碱金属/碱土金属、IB-IIB贵金属、VIII金属，还有三个Mn、In、V、Ti不好处理
    df = metal_type(df)
    # 1-3.私有 计算金属能量最大的轨道
    df = metal_max_energy_orbital(df)
    # 1-4.私用 计算金属能量最高的轨道的空轨道
    df['max_energy_empty_orbitals'] = df.apply(calculate_max_energy_empty_orbitals, axis=1)

    # 1-5.私有 计算每个金属的外层电子排布向量
    df = metal_outer_elec_config(df)
    # 1-6.私有 计算每个金属的外层电子空轨道数
    df = metal_outer_elec_empty_orbitals(df)

    df_rename_dict = {
        'max_energy_orbital':'max_metal_energy_orbital',
        'max_energy_empty_orbitals':'max_metal_energy_empty_orbitals'
        }
    df = df.rename(columns=df_rename_dict)
    # print (df.columns.to_list())

    # 1-7.私有 计算各金属氧化态的离子排布，这里对df列留
    df_metal_ion = metal_ions_elec_config(df)
    # 1-8.私有 计算金属离子能量最大的轨道
    df_metal_ion = metal_max_energy_orbital(df_metal_ion)

    # 1-9.私用 计算金属离子能量最高的轨道的空轨道
    df_metal_ion['max_ion_energy_empty_orbitals'] = df_metal_ion.apply(calculate_max_energy_empty_orbitals, axis=1)

    # 1-10.私有 计算金属离子的外层电子排布向量
    df_metal_ion = metal_ions_outer_elec_config(df_metal_ion)
    # 1-11.私有 计算金属离子的外层电子空轨道数
    df_metal_ion = metal_ions_outer_elec_empty_orbitals(df_metal_ion)

    df_metal_ion_rename_dict = {
        'max_energy_orbital': 'max_ion_energy_orbital',
        'max_energy_empty_orbitals': 'max_ion_energy_empty_orbitals'
    }
    df_metal_ion = df_metal_ion.rename(columns=df_metal_ion_rename_dict)

    # print(df_metal_ion.columns.to_list())

    df_all_cols = ['mof', 'metal', 'linker1smi', 'oxidation_state', 'temperature', 'time', 'MOF_num', '1s', '2s', '3s', '4s', '5s', '6s', '7s', '2p', '3p', '4p', '5p', '6p', '3d', '4d', '5d', '6d', '4f', '5f', 'period', 'metal_category', 'max_metal_energy_orbital', 'max_metal_energy_empty_orbitals', 'ns_metal', 'np_metal', 'n_1p_metal', 'n_1d_metal', 'n_2f_metal', 'ns_metal_empty', 'np_metal_empty', 'n_1p_metal_empty', 'n_1d_metal_empty', 'n_2f_metal_empty']
    df_allow_cols = ['mof', 'metal', 'linker1smi', 'oxidation_state', 'temperature', 'time', 'MOF_num', 'period', 'metal_category', 'max_metal_energy_orbital', 'max_metal_energy_empty_orbitals', 'ns_metal', 'np_metal', 'n_1p_metal', 'n_1d_metal', 'n_2f_metal', 'ns_metal_empty', 'np_metal_empty', 'n_1p_metal_empty', 'n_1d_metal_empty', 'n_2f_metal_empty']

    df_metal_ion_all_cols = ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '2p', '3p', '4p', '5p', '6p', '3d', '4d', '5d', '6d', '4f', '5f', 'MOF_num', 'period', 'max_ion_energy_orbital', 'max_ion_energy_empty_orbitals', 'ns_ions', 'np_ions', 'n_1p_ions', 'n_1d_ions', 'n_2f_ions', 'ns_ions_empty', 'np_ions_empty', 'n_1p_ions_empty', 'n_1d_ions_empty', 'n_2f_ions_empty']
    df_metal_ion_allow_cols = ['MOF_num', 'max_ion_energy_orbital', 'max_ion_energy_empty_orbitals', 'ns_ions', 'np_ions', 'n_1p_ions', 'n_1d_ions', 'n_2f_ions', 'ns_ions_empty', 'np_ions_empty', 'n_1p_ions_empty', 'n_1d_ions_empty', 'n_2f_ions_empty']

    # 过滤存在于df中的列名
    df_allow_cols_copy = [col for col in df_allow_cols if col in df.columns]

    # 然后合并
    df_merged = pd.merge(df[df_allow_cols_copy], df_metal_ion[df_metal_ion_allow_cols], on='MOF_num', how='inner')

    # # Merge the dataframes
    # df_merged = pd.merge(df[df_allow_cols], df_metal_ion[df_metal_ion_allow_cols], on='MOF_num', how='inner')

    return df_merged


def main_metal_bak(df):
    # 1-1.公用 确定金属所属周期
    df = metal_period(df)
    # 1-2.公用 确定金属所属类型，稀土金属、碱金属/碱土金属、IB-IIB贵金属、VIII金属，还有三个Mn、In、V、Ti不好处理
    df = metal_type(df)

    # 1-3.私有 计算每个金属的外层电子排布向量
    df = metal_outer_elec_config(df)
    # 1-4.私有 计算每个金属的外层电子空轨道数
    df = metal_outer_elec_empty_orbitals(df)

    # 1-5.私有 计算各金属氧化态的离子排布，这里对df列留
    df_metal_ion = metal_ions_elec_config(df)

    # 1-6.私有 计算金属离子的外层电子排布向量
    df_metal_ion = metal_ions_outer_elec_config(df_metal_ion)
    # 1-7.私有 计算金属离子的外层电子空轨道数
    df_metal_ion = metal_ions_outer_elec_empty_orbitals(df_metal_ion)

    # print(df_metal_ion.columns.to_list())

    df_all_cols = ['mof', 'metal', 'linker1smi', 'oxidation_state', 'temperature', 'time', 'MOF_num', '1s', '2s', '3s', '4s', '5s', '6s', '7s', '2p', '3p', '4p', '5p', '6p', '3d', '4d', '5d', '6d', '4f', '5f', 'period', 'metal_category', 'ns_metal', 'np_metal', 'n_1p_metal', 'n_1d_metal', 'n_2f_metal', 'ns_metal_empty', 'np_metal_empty', 'n_1p_metal_empty', 'n_1d_metal_empty', 'n_2f_metal_empty']
    df_allow_cols = ['mof', 'metal', 'linker1smi', 'oxidation_state', 'temperature', 'time', 'MOF_num', 'period', 'metal_category', 'ns_metal', 'np_metal', 'n_1p_metal', 'n_1d_metal', 'n_2f_metal', 'ns_metal_empty', 'np_metal_empty', 'n_1p_metal_empty', 'n_1d_metal_empty', 'n_2f_metal_empty']

    df_metal_ion_all_cols = ['1s', '2s', '3s', '4s', '5s', '6s', '7s', '2p', '3p', '4p', '5p', '6p', '3d', '4d', '5d', '6d', '4f', '5f', 'MOF_num', 'period', 'ns_ions', 'np_ions', 'n_1p_ions', 'n_1d_ions', 'n_2f_ions', 'ns_ions_empty', 'np_ions_empty', 'n_1p_ions_empty', 'n_1d_ions_empty', 'n_2f_ions_empty']
    df_metal_ion_allow_cols = ['MOF_num', 'ns_ions', 'np_ions', 'n_1p_ions', 'n_1d_ions', 'n_2f_ions', 'ns_ions_empty', 'np_ions_empty', 'n_1p_ions_empty', 'n_1d_ions_empty', 'n_2f_ions_empty']

    # 过滤存在于df中的列名
    df_allow_cols_copy = [col for col in df_allow_cols if col in df.columns]

    # 然后合并
    df_merged = pd.merge(df[df_allow_cols_copy], df_metal_ion[df_metal_ion_allow_cols], on='MOF_num', how='inner')

    # # Merge the dataframes
    # df_merged = pd.merge(df[df_allow_cols], df_metal_ion[df_metal_ion_allow_cols], on='MOF_num', how='inner')

    return df_merged


# df = main_metal(df)
# df.to_excel('1.xlsx')
# print (df)
# print (df[df.columns.difference(electron_config_columns + ['linker1smi','mof'])])