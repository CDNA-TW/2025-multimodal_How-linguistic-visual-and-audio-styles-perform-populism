import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_emotion(data_path,n_cluster,cluster_columns,valence,arousal,suffix,save_path):
    df = pd.read_csv(data_path)

    if suffix == "mean":
        # 平移valence_mean、arousal_mean至[-1,1]
        df[valence] = df[valence].apply(lambda x : x*2.0-1.0)
        df[arousal] = df[arousal].apply(lambda x : x*2.0-1.0)



    # 轉換欄位為數字類別索引
    c_labels, c_values = pd.factorize(df[cluster_columns])

    # 繪製散點圖，使用 c col 的不同類別顯示不同顏色
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(df[valence],df[arousal], c=c_labels, cmap="tab10", edgecolors="k")

    # 添加顏色條
    cbar = plt.colorbar(sc)
    cbar.set_label("clusters")

    # 顯示類別標籤對應關係
    unique_labels = list(set(zip(c_labels, df[cluster_columns])))
    legend_labels = {label: category for label, category in unique_labels}

    # 顯示圖例
    for label, category in legend_labels.items():
        plt.scatter([], [], c=[sc.to_rgba(label)], label=category, edgecolors="k")

    plt.legend(title="clusters")

    # 添加標籤
    plt.xlabel(f"valence_{suffix}")
    plt.ylabel(f"arousal_{suffix}")
    plt.title(f"valence-arousal distribution with {n_cluster} clusters")

    plt.axhline(0, color='black', linewidth=1)  # x 軸
    plt.axvline(0, color='black', linewidth=1)  # y 軸

    plt.savefig(save_path)
    # 顯示圖表
    plt.show()
    plt.clf()

def main(data_path,n_cluster,cluster_columns,mean,std):
    plot_emotion(data_path,n_cluster,cluster_columns,mean["col_name"][0],mean["col_name"][1],"mean",mean["save_path"])

    plot_emotion(data_path,n_cluster,cluster_columns,std["col_name"][0],std["col_name"][1],"std",std["save_path"])




if __name__ == ("__main__"):





    output_root = os.environ.get("OUTPUT_ROOT", "./output/analysis")
    setting = {
    "data_path" : os.environ.get("FEATURES_CSV", "./data/f1_f2_features_by_videos.csv"),
    "n_cluster" : "8",
    "cluster_columns" : "kmeans_cluster_8",
    "mean" : {"col_name" : ["valence_mean","arousal_mean"],
              "save_path" : os.path.join(output_root, "f12_8c_mean")},
    "std" : {"col_name" : ["valence_std","arousal_std"],
             "save_path" : os.path.join(output_root, "f12_8c_std")},
    }
    
    main(**setting)
