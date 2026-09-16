import os
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# SciPy: 用於階層式聚類
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster

# ------------------------------
# 1) 資料前處理
# ------------------------------
def prepare_data(drop_columns, df, id, marker, marker_columns):
    """
    1. 選取 numeric 欄位
    2. 移除指定的 drop_columns
    3. 若有 NaN，印出位置
    4. 取出 'id' 作為樣本名稱
    5. 將 'win' 對應到 win_marker，作為散佈圖的 marker
    """
    data = df.drop(columns=id).select_dtypes(include=[np.number])
    data = data.drop(drop_columns, axis=1)

    # 如果有缺失值，檢查並列出位置
    has_nan = np.isnan(data.to_numpy()).any()
    if has_nan:
        nan_indices = np.where(np.isnan(data))
        print("Indices of NaN values:", nan_indices)

    names = df[id].to_list()
    win_markers = [marker[x] for x in df[marker_columns]]

    return data, names, win_markers

# ------------------------------
# 2) PCA 降維至 2 維
# ------------------------------
def PCA_for_plotting(data):
    pca = PCA(n_components=2)
    data_reduced = pca.fit_transform(data)
    return data_reduced

# ------------------------------
# 3) 階層式分群 (4 種 linkage) + 繪圖
# ------------------------------
def hierarchical_clustering(
    data, 
    data_reduced, 
    names, 
    markers, 
    out_dir, 
    k_range=range(2, 13)
):
    """
    對給定的 data，依照 k_range 做迴圈，每個 k:
      1) 用 4 種 linkage (single, complete, average, ward) 分別分群
      2) 畫一張圖，含 4 個子圖 (subplot)，每個子圖是一種 linkage 結果的 2D scatter
      3) 計算 silhouette score 並存起來
      4) 回傳 cluster_labels 結果 (方便寫回 df)
    最後額外處理 k=1: 為每一種 linkage 繪製整棵 dendrogram，並用至少 10 種顏色標示。
    """

    os.makedirs(out_dir, exist_ok=True)

    # 我們要使用的 4 種 linkage 方法
    linkage_methods = ["single", "complete", "average", "ward"]

    # 用於儲存 [method -> { k -> labels }]
    cluster_results = {m: {} for m in linkage_methods}

    # 用於儲存 [method -> { k -> silhouette_score }]
    silhouette_dict = {m: {} for m in linkage_methods}

    # 預先算好距離向量 (pdist)
    dists = pdist(data)  # 預設 'euclidean'，ward 需要歐幾里得

    # -------------------------------
    # (A) 針對 k_range (預設 2~12)
    # -------------------------------
    for k in k_range:
        print(f"🔹 Hierarchical Clustering for K={k}")
        # 建立一張大圖，內含 4 個子圖
        fig, axes = plt.subplots(2, 2, figsize=(20, 15))
        fig.suptitle(f"Hierarchical Clustering (k={k})", fontsize=16)
        axes = axes.flatten()  # 轉成一維，axes[0]~axes[3]

        for i, method in enumerate(linkage_methods):
            # 1) 做 linkage
            Z = sch.linkage(dists, method=method)

            # 2) fcluster => 拿到分群結果
            #    用 criterion='maxclust'，代表要切成 k 群
            labels = fcluster(Z, k, criterion='maxclust')
            cluster_results[method][k] = labels

            # 3) 計算 silhouette score (若 k>1)
            if len(set(labels)) > 1:
                sil_score = silhouette_score(data, labels)
            else:
                sil_score = np.nan
            silhouette_dict[method][k] = sil_score

            # 4) 繪製 2D scatter plot
            ax = axes[i]
            df_plot = pd.DataFrame(data_reduced, columns=['x', 'y'])
            df_plot['cluster'] = labels
            df_plot['name'] = names
            df_plot['marker'] = markers

            sns.scatterplot(
                data=df_plot, x='x', y='y', hue='cluster', style='marker',
                style_order=['o', 'X'], palette='tab20', s=80, ax=ax
            )

            # 把樣本名稱標在點旁
            for j, row in df_plot.iterrows():
                ax.text(row['x'], row['y'], row['name'], fontsize=3, ha='left')

            ax.set_title(f"Linkage: {method}")

        # 儲存當前這個 k 的 4 種連結法對比圖
        out_path = os.path.join(out_dir, f"hierarchical_k_{k}.png")
        plt.savefig(out_path)
        plt.close()

    # -------------------------------
    # -------------------------------
    # (B) 在此處理 k=1: 輸出 dendrogram，但實際上切成 10 群並顯示不同顏色
    # -------------------------------
    n_samples = data.shape[0]

    for method in linkage_methods:
        Z = sch.linkage(dists, method=method)

        # 1. 計算使得分成10群的 threshold
        #    Z 排序依合併距離從小到大，若要 k=10，則取 Z[-(10-1), 2]
        #    (也就是「自底向上合併到剩10群」時的距離)
        if n_samples > 10:
            threshold = Z[-(10-1), 2]
        else:
            # 若 n_samples <= 10，理論上可以直接用 0 當 threshold
            # 但這情況下，每個樣本都可能是自己的群
            threshold = 0

        plt.figure(figsize=(10, 6))
        plt.title(f"Dendrogram (10 clusters) - Linkage: {method}")

        # 2. 呼叫 dendrogram，設定 color_threshold=threshold
        sch.dendrogram(
            Z,
            color_threshold=threshold,       # 在此距離切成10群
            above_threshold_color="gray",    # 超過 threshold 的分支(合併更多)都塗灰
            no_labels=False                  # 保留葉子標籤(若你的資料量很大，可視需要關掉)
        )
        plt.xlabel("Samples (or merged clusters)")
        plt.ylabel("Distance")

        dendro_path = os.path.join(out_dir, f"hierarchical_10clusters_{method}.png")
        plt.savefig(dendro_path)
        plt.close()


    # -------------------------------
    # (C) 繪製各 linkage 的 Silhouette Score 與 k 的趨勢圖
    # -------------------------------
    plt.figure(figsize=(8, 6))
    for method in linkage_methods:
        ks = sorted(silhouette_dict[method].keys())
        scores = [silhouette_dict[method][x] for x in ks]
        plt.plot(ks, scores, marker='o', label=method)
    plt.title("Silhouette Score vs. K (4 Linkage Methods)")
    plt.xlabel("Number of Clusters (K)")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "hierarchical_silhouette_scores.png"))
    plt.close()

    return cluster_results

# ------------------------------
# 4) main 函式
# ------------------------------
def main(
    drop_columns,
    data_path,
    id,
    marker,
    marker_columns,
    hierarchical_dir,    # 放置階層式分群輸出的資料夾
    out_path
):
    """
    主要程式入口：
    1. 讀取 CSV
    2. 產生 data, names, win_markers
    3. PCA 2D
    4. 呼叫階層式分群，針對 k=2~12 (可自行調整) 與 4 種 linkage
       -> 同時在最後處理 k=1 輸出 dendrogram (含 10色上色)
    5. cluster_results：把結果合併進 df 後輸出 CSV
    """

    df = pd.read_csv(data_path)

    # -- 1) 準備資料 --
    data, names, markers = prepare_data(drop_columns, df, id, marker, marker_columns)
    # 確保 df 的 index 與 data 保持一致
    df = df.loc[data.index].copy()

    # -- 2) PCA for plot --
    data_reduced = PCA_for_plotting(data)

    # -- 3) 階層式分群 (k=2~12 + dendrogram for k=1) --
    cluster_results = hierarchical_clustering(
        data, data_reduced, names, markers,
        out_dir=hierarchical_dir, 
    )

    # cluster_results 的結構： {method -> {k -> labels}}
    # e.g., cluster_results["single"][2] = [ ... labels ... ]
    #       cluster_results["complete"][2] = [ ... labels ... ]

    # -- 4) 寫回 df --
    # 為了和 k-means 的慣例相似，把每種 method + k 存成一欄
    for method, k_dict in cluster_results.items():
        for k, labels in k_dict.items():
            df[f"{method}_cluster_{k}"] = labels

    # -- 5) 輸出 CSV --
    df.to_csv(out_path, index=False)



if __name__ == "__main__":
    # 忽略 FutureWarning
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # 設定種子確保結果可重現
    np.random.seed(42)

    # 資料處理參數

    # markers = ['o', 's', 'D', '^', 'v']  # 圓形、正方形、菱形、上三角、下三角
    marker = {"L" : 'X',"W" : 'o'}
    marker_column = "win"
    # 共三個實驗可以用screen命令個別執行，或寫成迴圈
    # 實驗一 : 保留f1、f2且全體一起normalize
    data_path = os.environ.get(
        "AUDIO_FEATURES_CSV",
        "./data/ig_audios_features_with_election_result.csv",
    )

    row_path = os.environ.get("OUTPUT_ROOT", "./output/clustering/hierarchical_all")
    drop_columns = []

    id = "name"

    # 分群結果存取路徑
    hierarchical_dir = os.path.join(row_path,'hierarchical_fig')
    
    # 輸出最終結果的位置
    out_path = os.path.join(row_path,'hierarchical_output_ig_0901-1130.csv')

    main(drop_columns, data_path, id, marker,marker_column, hierarchical_dir, out_path)
