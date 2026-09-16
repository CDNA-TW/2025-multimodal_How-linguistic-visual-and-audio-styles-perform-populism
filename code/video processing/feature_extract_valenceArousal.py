import cv2
import os
import shutil
import pandas as pd
from tqdm import tqdm, trange
import time
from glob import glob
from collections import defaultdict
import numpy as np


def merge_files_by_prefix(folder_path, output_folder, file_extension="csv"):
    """Merge files with the same prefix in the given folder."""
    os.makedirs(output_folder, exist_ok=True)

    folder_path = glob(folder_path)

    # Group files by prefix
    prefix_to_files = defaultdict(list)
    for file_path in folder_path:
        filename = os.path.basename(file_path)
        prefix = "_".join(filename.split("_")[:5])
        prefix_to_files[prefix].append(file_path)

    processed_files = set()

    # Process files
    for prefix, files in prefix_to_files.items():
        if len(files) > 1:
            # Prioritize files with "cut" in the name
            cut_files = [f for f in files if "_cut_" in f]
            if cut_files:
                print(f"Files with 'cut' for prefix {prefix}: {cut_files}")
                representative_file = cut_files[0]  # Use the first "cut" file
                output_file = os.path.join(output_folder, os.path.basename(representative_file))
                shutil.copy(representative_file, output_file)
                print(f"Copied representative 'cut' file: {output_file}")
            else:
                # Merge all files for this prefix
                merged_df = pd.concat((pd.read_csv(file) for file in files), ignore_index=True)
                name = "_".join(files[0].split("_")[:-2]).split("/")[-1]
                output_file = os.path.join(output_folder, f"{name}_merged.{file_extension}")
                merged_df.to_csv(output_file, index=False)
                print(f"Merged and saved: {output_file}")
        else:
            # Copy single file directly
            single_file = files[0]
            output_file = os.path.join(output_folder, os.path.basename(single_file))
            shutil.copy(single_file, output_file)
            print(f"Copied single file: {output_file}")

        processed_files.update(files)

    print("File processing complete.")


def calculate_statistics(file, mappings):
    """Calculate statistics from the CSV file."""
    data = pd.read_csv(file)

    # Extract metadata from filename
    print(file)
    # account_name = os.path.basename(file).split("_")[3]
    if os.path.basename(file).split("_")[0] == "x":
        account_name = os.path.basename(file).split("_")[1] + " " +  os.path.basename(file).split("_")[2]
    else:
        account_name = os.path.basename(file).split("_")[0] + " " + os.path.basename(file).split("_")[1]

    # name = mappings["name_mapping"].get(account_name, account_name)
    name = account_name

    stats = {
        "party": mappings["party_mapping"].get(name, "Unknown"),
        "state": mappings["state_mapping"].get(name, "Unknown"),
        # "dateName": f"{os.path.basename(file).split('_')[0]} {account_name}",
        # "account_name": account_name,
        "name": name,
        "keycode":os.path.basename(file).replace("_va.csv", "")[-11:]
    }

    # Check for required columns
    emotion_columns = ["Neutral", "Happy", "Sad", "Surprise", "Fear", "Disgust", "Anger", "Contempt"]
    if not all(col in data.columns for col in ["emotion", "valence", "arousal"]):
        print(f"File {file} missing required columns. Skipping.")
        return None

    # Emotion distribution
    emotion_distribution = data["emotion"].value_counts(normalize=True).to_dict()
    stats.update({emotion: emotion_distribution.get(emotion, 0) for emotion in emotion_columns})

    # Valence and arousal statistics
    stats["valence_mean"] = data["valence"].mean()
    stats["valence_std"] = data["valence"].std()
    stats["arousal_mean"] = data["arousal"].mean()
    stats["arousal_std"] = data["arousal"].std()
    cal_data_df = pd.DataFrame([stats])
    
    expected_keys = ["party", "state", "name", "keycode",
                     "valence_mean", "valence_std", "arousal_mean", "arousal_std"] + emotion_columns

    for key in expected_keys:
        stats.setdefault(key, np.nan)

    return stats

if __name__ == "__main__":
    # root should point at the folder produced by visual_valence_arousal_loop.py
    # (its --output_csv argument, suffixed "_oneSecond" when --oneSecond is used).
    root = os.environ.get("PROJECT_ROOT", ".")
    input_folder = os.path.join(root, "ig_videos_va_output_csv_oneSecond","*va.csv")
    output_folder = os.path.join(root, "valenceArousal_ig_videos_output_csv_oneSecond/")
    os.makedirs(output_folder,exist_ok=True)

    # merge_files_by_prefix(folder_path=input_folder, output_folder=output_folder)

    files = glob(os.path.join(output_folder, "*va.csv")) # + glob(os.path.join(output_folder, "*merged.csv"))


    # Define mappings
    mappings = {
        "name_mapping": {
            'berniemorenoforohio': 'Bernie Moreno',
            'senbobcasey': 'Bob Casey',
            'bobcaseyjr': 'Bob Casey',
            'colinallred': 'Colin Allred',
            'repcolinallred': 'Colin Allred',
            'davemccormickpa': 'Dave McCormick',
            'senatorjontester': 'Jon Tester',
            'karilake': 'Kari Lake',
            'gallegoforaz': 'Ruben Gallego',
            'sensherrodbrown': 'Sherrod Brown',
            'podsaveamerica': 'Sherrod Brown',
            'sherrod': 'Sherrod Brown',
            'sentedcruz': 'Ted Cruz',
            'sheehyformt': 'Tim Sheehy',
        },

        "state_mapping": {
            'Kari Lake': 'Arizona',
            'Ruben Gallego': 'Arizona',
            'Tim Sheehy': 'Montana',
            'Jon Tester': 'Montana',
            'Bernie Moreno': 'Ohio',
            'Sherrod Brown': 'Ohio',
            'Bob Casey': 'Pennsylvania',
            'Dave McCormick': 'Pennsylvania',
            'Colin Allred': 'Texas',
            'Ted Cruz': 'Texas',
            'Tammy Baldwin':'Wisconsin',
            'Eric Hovde':'Wisconsin',
            'Mike Rogers':'Michigan',
            'Elissa Slotkin':'Michigan',
        },

        "party_mapping": {
            'Kari Lake': 'Republican',
            'Ruben Gallego': 'Democratic',
            'Tim Sheehy': 'Republican',
            'Jon Tester': 'Democratic',
            'Bernie Moreno': 'Republican',
            'Sherrod Brown': 'Democratic',
            'Bob Casey': 'Democratic',
            'Dave McCormick': 'Republican',
            'Colin Allred': 'Democratic',
            'Ted Cruz': 'Republican',
            'Mike Rogers': 'Republican',
            'Elissa Slotkin':'Democratic',
            'Tammy Baldwin':'Democratic',
            'Eric Hovde':'Republican',
        },
    }

    all_statistics = []

    for file in tqdm(files, desc="Processing files for statistics"):
        stats = calculate_statistics(file, mappings)
        all_statistics.append(pd.DataFrame([stats]))
    all_statistics_df = pd.concat(all_statistics, ignore_index=True)

    output_file = os.path.join(output_folder, "valence_arousal_0208.csv")
    all_statistics_df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Saved all statistics to {output_file}")


    # print(all_statistics_df)
    # calculate_statistics()