from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
from config import *
# 将SMILES字符串转化为分子对象
molecule = Chem.MolFromSmiles('c1nnc[nH]1')

# 计算分子指纹
fingerprint = AllChem.GetMorganFingerprintAsBitVect(molecule, 2)

# 将指纹转换为列表
fingerprint_list = list(fingerprint.ToBitString())

# # 打印结果
# print (fingerprint_list)
# print (len(fingerprint_list))



def example(df):
    from rdkit import Chem
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    # 使用RDKit库中的Chem模块读取特征文件中的linker数据
    # read the linker1 smiles and construct the molecule as mol1
    df["mol1"] = df["linker1smi"].apply(lambda smi: Chem.MolFromSmiles(smi))
    # print (df)
    x_unscaled_fp1 = np.array(
        [
            Chem.RDKFingerprint(mol1, maxPath=5, fpSize=512)
            for mol1 in df["mol1"].tolist()
        ]
    ).astype(float)
    print(x_unscaled_fp1)
    x_scaler_fp1 = StandardScaler()
    x_fp1 = x_scaler_fp1.fit_transform(x_unscaled_fp1)
    print(x_fp1)

def get_fingerprint(data):
    # Convert SMILES to RDKit molecule objects
    data['mol'] = data['linker1smi'].apply(Chem.MolFromSmiles)

    # Calculate RDKit fingerprints
    data['fp'] = data['mol'].apply(AllChem.GetMorganFingerprintAsBitVect, args=(2,))

    # Convert fingerprints to numpy arrays and stack them into a 2D array
    fps = np.stack(data['fp'].apply(lambda fp: np.array(fp)).to_numpy())

    # Add the fingerprint features to the dataframe
    data = pd.concat([data, pd.DataFrame(fps, columns=[f'fp{i}' for i in range(fps.shape[1])])], axis=1)




import re
# Function to find unique non-carbon parts in SMILES strings
def find_unique_non_carbon_parts(smiles_list):
    unique_parts = set()
    for smiles in smiles_list:
        # Find all non-carbon parts
        parts = re.findall(r'[^cC][^cC]*', smiles)
        # Add each part to the set of unique parts
        for part in parts:
            unique_parts.add(part)
    return unique_parts

def determine_linker1smi(df):
    # Find unique non-carbon parts in the 'linker1smi' column
    unique_non_carbon_parts_new = find_unique_non_carbon_parts(df['linker1smi'])

    # Sort the unique non-carbon parts by length in descending order
    unique_non_carbon_parts_new = sorted(unique_non_carbon_parts_new, key=len, reverse=True)

    # Define the known functional groups in descending order of length
    known_functional_groups = ['O=C(O)', 'P(=O)(O)O', 'O=P(O)(O)', 'P(=O)(O)O)', '[nH]', 'C=C3', 'C=C2', 'COO', 'CO',
                               'N', 'O=S(=O)(O)', '[O-]', 'N#', 'O=S1', 'ON1', 'NNN', 'S', '[O]', 'O)', 'O)O', '[N]',
                               'O=S(=O)(O)O', 'O=S1', 'N(', 'N1', 'NN', 'Nn1', 'O', 'l', 'n', 'nn', 'n[nH]', 'nnn1)N']

    # 含硫元素是这样的，除了硫酸基团以双键来判断配位数以外，其它的主要是硅，Si提供0个电子，CSC是两个孤对电子。目前还不明确的是s1、ns，这是不是-SH基团
    known_functional_groups = ['O=S(=O)(O)O', 'O=S(=O)(O)', 'S(=O)(=O)', 'O=S1', 'Si' , 'CSC'
                               ,'','','','',''
                               ]

    # 看下基团确定配位数：
    bind_count_dict = {
        '[O]':1,
        '(O)':1,
        '(=O)':1,
        'CSC':1,


    }


    known_functional_groups = sorted(known_functional_groups, key=len, reverse=True)


    # Find the non-carbon parts that are known functional groups
    known_parts = [part for part in unique_non_carbon_parts_new if part in known_functional_groups]

    # Find the non-carbon parts that are not known functional groups
    unknown_parts = [part for part in unique_non_carbon_parts_new if part not in known_functional_groups]
    for i in known_parts:
        if 'S' in known_parts:
            print(i)
    print ('='*100)
    for i in unknown_parts:
        if 'S' in unknown_parts:
            print(i)

    # Find the non-carbon parts that are not known functional groups
    # print(known_parts,'\n', unknown_parts)


# 最好的方向
# determine_linker1smi(df)
def get_pattern_best():
    mocule = r'O=C(O)C1=CCC(=S)N=C1'
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    # Define the molecule
    mol = Chem.MolFromSmiles(mocule)

    # 芳香环
    # Get the number of aromatic rings
    num_aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)

    print(f'The molecule has {num_aromatic_rings} aromatic rings.')

    # 苯环和吡啶环
    # Define the SMARTS patterns for benzene and pyridine
    benzene_smarts = 'c1ccccc1'
    pyridine_smarts = 'c1ccccn1'

    # Convert the SMARTS patterns to molecule objects
    benzene = Chem.MolFromSmarts(benzene_smarts)
    pyridine = Chem.MolFromSmarts(pyridine_smarts)

    # Get the number of benzene and pyridine rings
    num_benzene_rings = len(mol.GetSubstructMatches(benzene))
    num_pyridine_rings = len(mol.GetSubstructMatches(pyridine))

    print(f'The molecule has {num_benzene_rings} benzene rings and {num_pyridine_rings} pyridine rings.')


    # 羧基
    # Define a SMARTS pattern for the carboxyl group
    carboxyl_smarts = '[#6](=[#8])-[#8]'

    # Convert the SMARTS pattern to a molecule object
    carboxyl_pattern = Chem.MolFromSmarts(carboxyl_smarts)

    # Use the GetSubstructMatches method to find all matches
    matches = mol.GetSubstructMatches(carboxyl_pattern)

    # Print the number of matches
    print(f"Found {len(matches)} carboxyl groups.")

    # 定义官能团及其SMARTS表示的字典
    functional_groups = {
        'carboxyl': '[#6](=[#8])-[#8]',  # 羧基，由一个碳原子通过双键连接一个氧原子，并通过单键连接另一个氧原子构成
        # 含硫的
        'sulfonic_acid': '[#16](=[#8])(-[#8])-[#8]', # 磺酸基，描述的是磺酸（sulfonic acid,"O=S(=O)(O)O" ）官能团
        'sulfate': '[#16](=[#8])(=[#8])(-[#8])-[#8]',  # 硫酸基，由一个硫原子连接四个氧原子构成，其中两个氧原子通过双键连接
        'thiol': '[#16H]',  # 硫醇基，由一个硫原子连接一个氢原子构成
        'sulfide': '[#16]-[#6]',  # 硫化物，由一个硫原子连接一个碳原子构成
        'sulfone': '[#16](=[#8])=[#8]',  # 砜基，由一个硫原子连接两个氧原子构成，两个氧原子都通过双键连接
        'thiourea': '[#16](=[#8])-[#7]',  # 硫脲基，由一个硫原子连接一个氧原子和一个氮原子构成，硫与氧之间是双键
        'double_bonded_sulfur': '[#16]=[#6]',  # 双键连接的硫原子和碳原子，代表了C=S官能团

        # 含磷的
        'phosphate': '[#15](=[#8])(-[#8])-[#8]',  # 磷酸基，由一个磷原子连接三个氧原子构成，其中一个氧原子通过双键连接
        'organic_phosphorus': '[#15]',
        'phosphonate': '[#15](-[#8])-[#8]',
        'double_bond_organic_phosphorus': '[#15]=[#6]',

        # 含溴的
        'bromine': '[#35]',  # 溴原子

    }

    # 遍历字典中的每一个官能团
    for group_name, smarts in functional_groups.items():
        # 将SMARTS模式转换为分子对象
        pattern = Chem.MolFromSmarts(smarts)
        # 使用GetSubstructMatches方法找到所有的匹配
        matches = mol.GetSubstructMatches(pattern)
        # 打印匹配的数量
        print(f"在分子中找到了 {len(matches)} 个 {group_name} 官能团。")

def cal_patter_cnt():
    from rdkit import Chem
    import pandas as pd

    # 定义官能团的SMARTS模式
    functional_groups = {
        'sulfonic_acid': '[#16](=[#8])(-[#8])-[#8]',  # 磺酸基
        'sulfate': '[#16](=[#8])(=[#8])(-[#8])-[#8]',  # 硫酸基
        'thiol': '[#16H]',  # 硫醇基
        'sulfide': '[#16]-[#6]',  # 硫化物
        'sulfone': '[#16](=[#8])=[#8]',  # 砜基
        'thiourea': '[#16](=[#8])-[#7]',  # 硫脲基
        'double_bonded_sulfur': '[#16]=[#6]',  # 双键连接的硫原子和碳原子
    }

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

    # 创建一个新的DataFrame来存储计数
    counts_df = pd.DataFrame()

    # 遍历DataFrame中的分子
    for idx, row in df.iterrows():
        smi = row['linker1smi']
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            counts = count_smarts_patterns(mol, functional_groups)
            counts_df = counts_df.append(counts, ignore_index=True)

    print (counts_df)



# Load the data again
data_new = df

# Extract all unique elements from 'linker1smi' column
elements = set()

# Function to add unique elements to the set
def add_elements(smiles):
    for char in re.findall(r'[A-Z][a-z]*', smiles):
        elements.add(char)

# Apply function to each row in 'linker1smi'
data_new['linker1smi'].apply(add_elements)

# Display unique elements
elements = elements - {'C', 'N', 'O', 'S', 'P', 'Si', 'H'}
print (elements)
