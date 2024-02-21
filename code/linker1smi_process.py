from metal_elec_config import *
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AtomPairs import Torsions
from rdkit.Avalon import pyAvalonTools as pyav
import numpy as np
import re

# 这里帮我们判断有什么元素
def assist_get_all_elements(df):
    # Extract all unique elements from 'linker1smi' column
    elements = set()

    # Function to add unique elements to the set
    def add_elements(smiles):
        for char in re.findall(r'[A-Z][a-z]*', smiles):
            elements.add(char)

    # Apply function to each row in 'linker1smi'
    df['linker1smi'].apply(add_elements)

    # Display unique elements
    elements = elements - {'C', 'N', 'O', 'S', 'P', 'Si', 'H','F', 'Cl', 'Br', 'I'}
    print(elements)

def count_smarts_patterns(mol, smarts_dict):
    counts = {}
    for name, smarts in smarts_dict.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is not None:
            matches = mol.GetSubstructMatches(pattern)
            counts[name] = len(matches)
        else:
            counts[name] = 0
    return counts

def add_functional_groups(df):
    # 定义官能团的SMARTS模式
    # H、C、N、O、S、P、Si、F、Cl、Br、I 芳香环
    functional_groups = {
        # 含氢的，这里思前想后，不用了
        # 'methyl_group': '[CH2]',  #
        # 'ethynyl_group': '[CH]',  #
        # 'amino group': '[NH]',  #
        # 'phosphino_group': '[PH]',  # 磷醇基，与下面的重复了

        # 含碳的
        'alkene': '[#6]=[#6]',  # 碳碳双键
        'acetylene': '[#6]#[#6]',  # 碳碳三键

        # 含氮的
        'amine': '[#7]([#1])',  # 胺基，由一个氮原子连接一个氢原子
        'primary_amide': '[#6](=[#8])-[#7]',  # 一级酰胺，由一个碳原子通过双键连接一个氧原子，并通过单键连接一个氮原子，该氮原子还通过单键连接一个氢原子构成
        'urea': '[#6](=[#8])-[#7]([#1])-[#6](=[#8])',  # 尿素，由两个碳原子，每个碳原子都通过双键连接一个氧原子，两个碳原子通过一个氮原子连接构成
        'secondary_amide': '[#6](=[#8])-[#7](-[#6])-[#6]',  # 二级酰胺，由一个碳原子通过双键连接一个氧原子，并通过单键连接一个氮原子，该氮原子还通过单键连接两个碳原子构成
        'imine': '[#6]=[#7]',  # 脒，由一个碳原子通过双键连接一个氮原子构成
        'thiourea': '[#16](=[#8])-[#7]',  # 硫脲基，由一个硫原子连接一个氧原子和一个氮原子构成，硫与氧之间是双键
        'guanidine': '[#6](=[#7])-[#7]([#1])',  # 胍基，由一个碳原子通过双键连接一个氮原子，并通过单键连接一个氮原子，该氮原子还通过单键连接一个氢原子构成
        'amidino': '[#7]([#1])-[#6](=[#8])',  # 阿米基，由一个氮原子通过单键连接一个氢原子，并通过单键连接一个碳原子，该碳原子还通过双键连接一个氧原子构成

        # 含氧的
        'carboxyl': '[#6](=[#8])-[#8]',  # 羧基，由一个碳原子通过双键连接一个氧原子，并通过单键连接另一个氧原子构成
        'hydroxyl': '[#8]-[#1]',  # 羟基，由一个氧原子连接一个氢原子构成
        'carbonyl': '[#6]=[#8]',  # 醛基或酮基，由一个碳原子通过双键连接一个氧原子构成
        'ether': '[#6]-[#8]-[#6]',  # 醚基，由一个氧原子连接两个碳原子构成
        'ester': '[#6](=[#8])-[#8]-[#6]',  # 酯基，由一个碳原子通过双键连接一个氧原子，并通过单键连接另一个氧原子，该氧原子还通过单键连接一个碳原子构成
        'peroxide': '[#8]-[#8]',  # 过氧化物，由两个氧原子通过单键连接构成
        'acid_anhydride': '[#6](=[#8])-[#8]-[#6](=[#8])',  # 酸酐，由两个碳原子，每个碳原子都通过双键连接一个氧原子，两个碳原子通过一个氧原子连接构成

        # 含硫的
        'sulfonic_acid': '[#16](=[#8])(-[#8])-[#8]',  # 磺酸基，由一个硫原子连接三个氧原子构成，其中一个氧原子通过双键连接
        'sulfate': '[#16](=[#8])(=[#8])(-[#8])-[#8]',  # 硫酸基，由一个硫原子连接四个氧原子构成，其中两个氧原子通过双键连接
        'thiol': '[#16H]',  # 硫醇基，由一个硫原子连接一个氢原子构成
        'sulfide': '[#16]-[#6]',  # 硫化物，由一个硫原子连接一个碳原子构成
        'sulfone': '[#16](=[#8])=[#8]',  # 砜基，由一个硫原子连接两个氧原子构成，两个氧原子都通过双键连接
        'double_bond_sulfur_carbon': '[#16]=[#6]',  # 硫碳双键，代表了C=S官能团
        'CSC': '[#6]-[#16]-[#6]',  # CSC基团
        'disulfide': '[#16]-[#16]',  # 二硫键，由两个硫原子通过单键连接构成

        # 含磷的
        'diphosphate': '[#15](=[#8])(-[#8])(-[#8])-[#6]',  # 磷酸二酯
        'phosphate': '[#15](=[#8])(-[#8])-[#8]',  # 磷酸基，P(=O)(O)O)，由一个磷原子连接三个氧原子构成，其中一个氧原子通过双键连接
        'phosphate_ester': '[#15](-[#8])-[#8]',  # 磷酸酯
        'phosphonate': '[#15](=[#8])-[#6]',  # 磷酰基
        'organo_phosphate': '[#15](-[#8])-[#6]',  # 有机磷酸酯
        'double_bond_organic_phosphorus': '[#15]=[#6]',  # 双键连接的有机磷
        'phosphorus_hydrogen_bond': '[#15]-[#1]',  # 磷氢键，这个很少见哦
        'organic_phosphorus': '[#15]',  # 有机磷

        # 含硅的
        'silicon': '[#14]',  # Si

        # 含卤素的，我觉得F不能和其它的放一起
        'fluorine': '[#9]',  # 氟原子
        # 'chlorine': '[#17]',  # 氯原子
        # 'bromine': '[#35]',  # 溴原子
        # 'iodine': '[#53]',  # 碘原子
        'halogens': '[#17,#35,#53]',  # 包含任意一种卤素的模式
    }

    # functional_ring_groups = {
    #     'benzene': 'c1ccccc1',  # 苯环
    #     'pyridine': 'n1ccccc1',  # 吡啶环
    #     'pyrimidine': 'n1ccncc1',  # 嘧啶环
    #     'imidazole': 'n1ccnc1',  # 咪唑环
    #     'thiophene': 's1cccc1',  # 噻吩环
    #     'phosphole': 'p1cccc1',  # 磷杂环
    #     'furan': 'o1cccc1',  # 呋喃环
    #     'pyrrole': 'n1cccc1',  # 吡咯环
    # }

    # 创建一个新的DataFrame来存储计数
    counts_df = pd.DataFrame()

    # Create an empty list to store counts dictionaries
    counts_list = []

    for idx, row in df.iterrows():
        smi = row['linker1smi']
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            counts = count_smarts_patterns(mol, functional_groups)
            # Add counts dictionary to the list
            counts_list.append(counts)

    # Convert list of dictionaries to DataFrame
    counts_df = pd.DataFrame(counts_list)
    # print (len(counts_df.columns))

    # Identify the columns where all values are zero
    cols_to_remove = counts_df.columns[counts_df.sum(axis=0) == 0]
    # cols_to_remove = ['amine', 'urea', 'thiourea', 'guanidine', 'amidino', 'hydroxyl','thiol', 'disulfide', 'double_bond_organic_phosphorus','phosphorus_hydrogen_bond']
    # print (cols_to_remove)
    # Drop these columns from the DataFrame
    counts_df = counts_df.drop(columns=cols_to_remove)

    # print (counts_df.columns.to_list())
    functional_groups_columns = ['alkene', 'acetylene', 'primary_amide', 'secondary_amide', 'imine', 'carboxyl', 'carbonyl', 'ether', 'ester', 'peroxide', 'acid_anhydride', 'sulfonic_acid', 'sulfate', 'sulfide', 'sulfone', 'double_bond_sulfur_carbon', 'CSC', 'diphosphate', 'phosphate', 'phosphate_ester', 'phosphonate', 'organo_phosphate', 'organic_phosphorus', 'silicon', 'fluorine', 'halogens']
    # print (counts_df)

    return counts_df

def add_aromatic_rings(df):
    aromatic_ring_groups = {
        'benzene': 'c1ccccc1',  # 苯环
        'pyridine': 'n1ccccc1',  # 吡啶环
        'pyrimidine': 'n1ccncc1',  # 嘧啶环
        'imidazole': 'n1ccnc1',  # 咪唑环
        'thiophene': 's1cccc1',  # 噻吩环
        'phosphole': 'p1cccc1',  # 磷杂环
        'furan': 'o1cccc1',  # 呋喃环
        'pyrrole': 'n1cccc1',  # 吡咯环
    }

    # 创建一个空列表来存储计数字典
    counts_list = []

    # 遍历DataFrame中的分子
    for idx, row in df.iterrows():
        smi = row['linker1smi']
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            counts = count_smarts_patterns(mol, aromatic_ring_groups)
            # 将counts字典添加到列表中
            counts_list.append(counts)

    # 将字典列表转换为DataFrame
    counts_df = pd.DataFrame(counts_list)

    counts_df['total_aromatic_rings'] = counts_df.sum(axis=1)

    # Identify the columns where all values are zero
    cols_to_remove = counts_df.columns[counts_df.sum(axis=0) == 0]
    # cols_to_remove = ['phosphole']
    # print (cols_to_remove)
    # Drop these columns from the DataFrame
    counts_df = counts_df.drop(columns=cols_to_remove)


    # print (counts_df.columns.to_list())
    aromatic_rings_columns = ['benzene', 'pyridine', 'pyrimidine', 'imidazole', 'thiophene', 'furan', 'pyrrole', 'total_aromatic_rings']
    # print(counts_df)
    return counts_df

def add_hetero_atoms(df):
    hetero_atoms = ['N', 'O', 'S', 'P', 'Si', 'F', 'Cl', 'Br', 'I']

    # 创建一个新的DataFrame来存储计数
    counts_df = pd.DataFrame(columns=hetero_atoms)

    # 创建一个空列表来存储计数字典
    counts_list = []

    # 遍历DataFrame中的分子
    for idx, row in df.iterrows():
        smi = row['linker1smi']
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            # 创建一个字典来存储每种杂原子的数量
            counts = {atom: 0 for atom in hetero_atoms}
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in counts:
                    counts[symbol] += 1
            # 将counts字典添加到列表中
            counts_list.append(counts)

    # 将字典列表转换为DataFrame
    counts_df = pd.DataFrame(counts_list)

    counts_df['total_N_O_P_S'] = counts_df[['N', 'O', 'P', 'S']].sum(axis=1)
    counts_df['total_Cl_Br_I'] = counts_df[['Cl', 'Br', 'I']].sum(axis=1)
    counts_df['total_hetero_atoms'] = counts_df.sum(axis=1)

    # Identify the columns where all values are zero
    cols_to_remove = counts_df.columns[counts_df.sum(axis=0) == 0]
    # cols_to_remove = ['phosphole']
    # print (cols_to_remove)
    # Drop these columns from the DataFrame
    counts_df = counts_df.drop(columns=cols_to_remove)

    # print (counts_df.columns.to_list())
    aromatic_rings_columns = ['N', 'O', 'S', 'P', 'Si', 'F', 'Cl', 'Br', 'I', 'total_N_O_P_S', 'total_Cl_Br_I', 'total_hetero_atoms']


    # print(counts_df)
    return counts_df

def cal_fingerprint(df):
    # Read the linker1 smiles and construct the molecule as mol1
    df["mol1"] = df["linker1smi"].apply(lambda smi: Chem.MolFromSmiles(smi))

    # Calculate different types of fingerprints for each molecule

    # 1. Morgan fingerprint
    df['morgan_fp'] = df['mol1'].apply(lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2))

    # 2. RDKit fingerprint
    df['rdkit_fp'] = df['mol1'].apply(lambda mol: Chem.RDKFingerprint(mol, maxPath=5, fpSize=512))

    # 3. Topological Torsion fingerprint
    # Calculate topological torsion fingerprints
    df['tt_fp'] = df['mol1'].apply(lambda mol: Torsions.GetTopologicalTorsionFingerprint(mol))
    # Convert LongSparseIntVect to dictionary
    df['tt_fp_dict'] = df['tt_fp'].apply(lambda fp: fp.GetNonzeroElements())

    # Convert the dictionary to a bit vector
    # Here, we assume that the size of the bit vector is 1024
    df['tt_fp_bit'] = df['tt_fp_dict'].apply(lambda fp_dict: [1 if i in fp_dict else 0 for i in range(1024)])


    # 4. Atom Pair fingerprint
    df['ap_fp'] = df['mol1'].apply(lambda mol: Chem.rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol))

    # 5. Avalon fingerprint
    # Calculate Avalon fingerprints
    df['avalon_fp'] = df['mol1'].apply(lambda mol: pyav.GetAvalonFP(mol))
    df = df.drop(columns=['mol1'])
    # Print the dataframe
    # print (df[['morgan_fp','rdkit_fp','tt_fp_bit','ap_fp','avalon_fp']])

    # # Calculate the length of each fingerprint
    # df['morgan_fp_len'] = df['morgan_fp'].apply(len)
    # df['rdkit_fp_len'] = df['rdkit_fp'].apply(len)
    # df['tt_fp_bit_len'] = df['tt_fp_bit'].apply(len)
    # df['ap_fp_len'] = df['ap_fp'].apply(len)
    # df['avalon_fp_len'] = df['avalon_fp'].apply(len)
    #
    # # Print the lengths
    # print(df[['morgan_fp_len', 'rdkit_fp_len', 'tt_fp_bit_len', 'ap_fp_len', 'avalon_fp_len']])

    # df.to_pickle('./1.pkl')
    return df

def main_linker(df):
    # 依次获取其官能团数量、芳香环数量、杂原子数量
    df = main_metal(df)
    # print (len(df.columns.to_list()),df.columns.to_list())
    counts_functional_groups_df = add_functional_groups(df)
    # print (len(counts_functional_groups_df.columns.to_list()),counts_functional_groups_df.columns.to_list())
    count_aromatic_rings = add_aromatic_rings(df)
    # print (len(count_aromatic_rings.columns.to_list()),count_aromatic_rings.columns.to_list())
    count_hetero_atoms = add_hetero_atoms(df)
    # print (len(count_hetero_atoms.columns.to_list()),count_hetero_atoms.columns.to_list())

    df = pd.concat([df, counts_functional_groups_df,count_aromatic_rings,count_hetero_atoms], axis=1)


    df = cal_fingerprint(df)
    # print (len(df.columns.to_list()),df.columns.to_list())
    # df.to_excel('1.xlsx')
    return df
    # print (df.columns.to_list())
    # print (df)
main_linker(df)