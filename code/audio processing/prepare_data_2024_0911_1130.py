import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

def standardize_columns(df,id):
    # 排除id，所有數值欄位
    columns_to_normalize = df.select_dtypes(include=np.number).columns.difference([id])
     # 初始化StandardScaler
    scaler = StandardScaler()
    # 對指定欄位進行標準化
    df[columns_to_normalize] = scaler.fit_transform(df[columns_to_normalize])
    return df

def fill_missing_values(df,id,fill_missing):
    # 使用KNN Imputer進行缺失值插補
    imputer = KNNImputer(n_neighbors=fill_missing)
    columns_to_imputed = df.select_dtypes(include=np.number).columns.difference([id])
    df_imputed = pd.DataFrame(imputer.fit_transform(df[columns_to_imputed]), columns=columns_to_imputed)
    # 將id欄位加回去，並保持原始順序
    df_imputed[id] = df[id].reset_index(drop=True)
    # 重新排序列，將id放在最前面
    cols = [id] + [col for col in df_imputed.columns if col != id]
    df_imputed = df_imputed[cols]
    return df_imputed


# 取得新的政黨欄位 : 
def get_new_col(df,id,party_dict):
    df["Name"] = df[id].apply(lambda x : "_".join(x.split("_")[:-1])[:-12])
    df["short_code"] = df[id].apply(lambda x : "_".join(x.split("_")[:-1])[-11:])
    df["party"] = ""
    for name in party_dict:
        df.loc[df["Name"].apply(lambda x : x.replace("_","")) == name,'party'] = party_dict[name]
    df = df.drop(id, axis=1)
    df = df[["Name","short_code","party"]+list(df.columns[:-3])]
    return df

def main(audio_path,output_path,id,fill_missing,party_dict):
    df = pd.read_csv(audio_path)
    print(df[id].head())
    print(df.columns[df.isna().any()])
    df = standardize_columns(df,id)
    if fill_missing > 0 :
        df = fill_missing_values(df,id,fill_missing)
    df = get_new_col(df,id,party_dict)
    
    # 驗證結果
    print(df.head(5))
    print(df.shape)
    print(df.columns[df.isna().any()])
    df.to_csv(output_path, index=False)
    

if __name__ == "__main__":
    # 欄位要有short_code和Name的對照
    #,States,Name,Party,post,footage,short_code,text,like,comment
    audio_path = os.environ.get(
        "AUDIO_FEATURES_OPENSMILE_CSV",
        "./data/ig_audios_features_opensmile.csv",
    )
    output_path = os.environ.get(
        "OUTPUT_CSV",
        "./output/ig_audios_features_opensmile_with_party.csv",
    )
    
    # id 是擷取檔名和相關資訊後，要輸出的欄位名稱
    # 原filename : Mike_Rogers_DA9tXO2vN4v_vocals
    # 輸出檔名如下 Mike_Rogers_DA9tXO2vN4v.csv
    # 要擷取的部分是Mike_Rogers(人名)，DA9tXO2vN4v(shortcode唯一編碼)
    id = "filename"

    fill_missing = 0
    

    # add_party是True才會生效
    party_dict = {
        "BernieMoreno" : "R",
        "BobCasey" : "D",
        "ColinAllred" : "D",
        "DaveMcCormick": "R",
        "KariLake" : "R",
        "RubenGallego" : "D",
        "SherrodBrown" : "D",
        "TedCruz" : "R",
        "MikeRogers" : "R",
        "ElissaSlotkin" : "D",
        "EricHovde" : "R",
        "TammyBaldwin" : "D",
    }

    main(audio_path,output_path,id,fill_missing,party_dict)