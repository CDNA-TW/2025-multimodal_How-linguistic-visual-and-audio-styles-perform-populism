import os
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.feature_selection import SelectKBest, f_classif
# import multiprocessing as mp
# from multiprocessing import Pool


def prepare_data(drop_columns, df, id, marker,marker_columns):
    # 轉換為 NumPy 陣列
    data = df.drop(columns= id).select_dtypes(include=[np.number])
    data = data.drop(drop_columns,axis=1)
    # 如果有缺失值，找到具體位置
    has_nan = np.isnan(data.to_numpy()).any()
    if has_nan:
        nan_indices = np.where(np.isnan(data))
        print("Indices of NaN values:", nan_indices)

    names = df[id].to_list()
    markers = [marker[x] for x in df[marker_columns]]
    return data,names,markers

def PCA_for_plotting(data):
    # 使用 PCA 降維至 2 維
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)
    return data_reduced

def k_means(data, data_reduced, names, markers,kmeans_dir, silhouette_path, elbow_path, k_range=range(2, 13), n_init=500):
    """
    執行 KMeans 並對 `k_range` 內的所有 K 值進行 clustering，
    存儲:
    - 每個 K 值的聚類結果於 dict，回傳給 main() 存入 df
    - 每個 K 值的散點圖至 `kmeans_dir`
    - Silhouette Score 圖
    - Elbow Method 圖
    """
    silhouette_scores = {}  # 改為字典來存不同 K 的 silhouette score
    elbow_scores = {}  # 存 K 對應的 SSE（Sum of Squared Errors）
    cluster_results = {}

    # 確保輸出目錄存在
    os.makedirs(kmeans_dir, exist_ok=True)
    os.makedirs(silhouette_path, exist_ok=True)
    os.makedirs(elbow_path, exist_ok=True)

    for k in k_range:
        print(f"🔹 執行 KMeans for K={k}")

        # 執行 KMeans
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=42).fit(data)
        cluster_labels = kmeans.labels_
        inertia = kmeans.inertia_

        # 儲存不同 K 的 clustering labels
        cluster_results[k] = cluster_labels

        # 計算 Silhouette Score
        if len(set(cluster_labels)) > 1:  # 確保 KMeans 至少產生 2 個不同群
            silhouette_scores[k] = silhouette_score(data, cluster_labels)
        else:
            silhouette_scores[k] = np.nan  # 只有一個 cluster，無法計算 Silhouette Score

        elbow_scores[k] = inertia  # 儲存 SSE（誤差平方和）

        # 建立 DataFrame 來整合繪圖資料
        df_plot = pd.DataFrame(data_reduced, columns=['x', 'y'])
        df_plot['cluster'] = cluster_labels
        df_plot['name'] = names
        df_plot['marker'] = markers

        # 繪製聚類圖
        plt.figure(figsize=(20, 15))
        ax = sns.scatterplot(data=df_plot, x='x', y='y', hue='cluster', style='marker',
                             style_order=['o', 'X'], palette='tab20', s=80)

        for i, row in df_plot.iterrows():
            ax.text(row['x'], row['y'], row['name'], fontsize=3, ha='left', fontweight='bold')

        plt.title(f'KMeans Clustering with {k} Clusters')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')

        # 儲存圖片到 `kmeans_dir`
        kmeans_path = os.path.join(kmeans_dir, f"kmeans_{k}.png")
        plt.savefig(kmeans_path)
        plt.close()

    # **修正：在迴圈結束後繪製 Silhouette Score 圖**
    plt.figure()
    plt.plot(list(silhouette_scores.keys()), list(silhouette_scores.values()), marker='o')
    plt.title('Silhouette Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join(silhouette_path, "silhouette_scores.png"))  # 統一存成 silhouette_scores.png
    plt.close()

    # **修正：在迴圈結束後繪製 Elbow Method 圖**
    plt.figure()
    plt.plot(list(elbow_scores.keys()), list(elbow_scores.values()), marker='o')
    plt.title('Elbow Method for Optimal K')
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.savefig(os.path.join(elbow_path, "elbow_scores.png"))  # 統一存成 elbow_scores.png
    plt.close()

    return cluster_results

def main(drop_columns, data_path, id, marker_columns,marker, kmeans_dir, silhouette_path, elbow_path, out_path):
    df = pd.read_csv(data_path)
    
    # 準備資料
    data, names, markers = prepare_data(drop_columns, df, id, marker,marker_columns)

    data_reduced = PCA_for_plotting(data)

    # 執行 KMeans，取得不同 K 值的 clustering labels
    kmeans_cluster_labels_dict = k_means(data, data_reduced, names, markers, kmeans_dir, silhouette_path, elbow_path)

    # **確保 `df` 只保留 `data` 的對應索引**
    df = df.loc[data.index].copy()

    # **將每個 K 值的 clustering labels 存入 `df`**
    for k, labels in kmeans_cluster_labels_dict.items():
        df[f"kmeans_cluster_{k}"] = labels  # 存為新欄位

    # 儲存結果
    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    # 忽略 FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 設定種子確保結果可重現
    np.random.seed(42)

    # 資料處理參數

    # markers = ['o', 's', 'D', '^', 'v']  # 圓形、正方形、菱形、上三角、下三角
    marker = {"L" : 'X',"W" : 'o'}
    marker_columns = "win"
    
    # 共三個實驗可以用screen命令個別執行，或寫成迴圈
    # 實驗一 : 保留f1、f2且全體一起normalize
    data_path = os.environ.get("FEATURES_CSV", "./data/basic_features_by_videos_raw.csv")
    row_path = os.environ.get("OUTPUT_ROOT", "./output/clustering/k-means_basic")
    drop_columns = []
    # 實驗二 : 移除f1、f2且全體一起normalize
    # data_path = "./data/f1_f2_features_by_videos_raw.csv"
    # row_path = "./output/clustering/k-means_f12"
    # drop_columns = []


    # id : 使用id對應的欄位作資料區別，畫圖的時候顯示在每筆資料上。
    id = "name"

    # 分群結果存取路徑
    kmeans_dir = os.path.join(row_path,'kmeans_fig')
    # kmeans方法的效果驗證
    silhouette_dir = os.path.join(row_path,'kmeans_silhouettey_score')
    elbow_dir = os.path.join(row_path,'kmeans_elbow_score')
    
    # 輸出最終結果的位置
    out_path = os.path.join(row_path,'kmeans_output_ig_0901-1130.csv')

    main(drop_columns, data_path, id, marker_columns,marker, kmeans_dir, silhouette_dir, elbow_dir, out_path)