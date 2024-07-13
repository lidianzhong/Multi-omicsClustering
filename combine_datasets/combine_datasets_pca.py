import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def combine_datasets_pca(data, n_components_list):
    """
    对给定的数据集应用 PCA 降维，并将它们拼接成一个数据集，同时保留样本名称信息。
    
    参数:
    - data: list of DataFrame, 输入数据集列表，每个DataFrame的列为样本，行为特征
    - n_components_list: list, 每个数据集对应的 PCA 降维后的维数列表
    
    返回:
    - combined_data_df: DataFrame, 拼接后的降维数据，包含样本名称信息
    """
    pca_transformed_data = []
    for dataset, n_components in zip(data, n_components_list):
        pca = PCA(n_components=n_components)
        # 假设dataset是DataFrame，其列名为样本名称
        transformed = pca.fit_transform(dataset.T)
        # 将PCA变换后的数据转换为DataFrame，并使用原始数据集的列名（样本名称）作为索引
        transformed_df = pd.DataFrame(transformed, index=dataset.columns)
        pca_transformed_data.append(transformed_df)
    
    # 使用pd.concat而不是np.concatenate来拼接数据，以保留索引（样本名称）
    combined_data_df = pd.concat(pca_transformed_data, axis=1)
    
    return combined_data_df