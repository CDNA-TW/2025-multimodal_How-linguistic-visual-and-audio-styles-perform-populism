import os
import pandas as pd
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from feat.detector import Detector

# Function to process a single video (initializes its own detector with CUDA)
def process_video(args):
    folder, save_root, save_folder = args
    try:
        # Initialize the detector (CUDA)
        detector_ = Detector(verbose=True, device='cuda')

        print(f"Processing: {folder}")
        out_name = os.path.join(save_root, save_folder, f"{os.path.basename(folder).split('.')[0]}.csv")

        if not os.path.exists(out_name):
            # Detect emotions and save
            fex = detector_.detect_video(folder, data_type="video", skip_frames=30, save=out_name, 
                                    num_workers = 4)
            fex.to_csv(out_name, index=False)
            print(f"{out_name} is saved!")

            # Calculate head pose change angles
            df = pd.read_csv(out_name)
            pitch_lst = df['Pitch'].tolist()
            roll_lst = df['Roll'].tolist()
            yaw_lst = df['Yaw'].tolist()

            for index, (p_angle, r_angle, y_angle) in enumerate(zip(pitch_lst, roll_lst, yaw_lst)):
                if index == 0:
                    continue
                df.loc[index, "pitch_gap"] = p_angle - pitch_lst[index - 1]
                df.loc[index, "roll_gap"] = r_angle - roll_lst[index - 1]
                df.loc[index, "yaw_gap"] = y_angle - yaw_lst[index - 1]

            # Save the modified DataFrame with gap calculations
            df.to_csv(out_name.replace('.csv', '_addGap.csv'), index=False)
        else:
            print(f"Skipping {folder} (already processed)")

    except Exception as e:
        print(f"Error processing {folder}: {e}")



def process_videos_in_parallel(video_list, num_workers, save_root, save_folder):
    args_list = [(video, save_root, save_folder) for video in video_list]

    with Pool(processes=num_workers) as pool:
        list(tqdm(pool.imap_unordered(process_video, args_list), total=len(video_list)))


def removeExistFile(all_folder, remove_folder):
    remove_filenames = [os.path.basename(path).replace("_cut.mp4", "") for path in remove_folder]
    filtered_all_folder = []

    for path in all_folder:
        filename = os.path.basename(path)
        print(f"Checking file: {filename}")  

        if not any(remove_name in filename for remove_name in remove_filenames):
            filtered_all_folder.append(path)
    
    return filtered_all_folder


if __name__ == "__main__":
    # Get the list of video files
    # Set these via environment variables, or edit the defaults below.
    root = os.environ.get("PROJECT_ROOT", ".")
    save_folder = "ig_videos_pyfeat_raw"
    os.makedirs(os.path.join(root, save_folder), exist_ok = True)

    video_glob = os.environ.get("VIDEO_GLOB", "./data/ig_videos_validRaw/*/*mp4")
    folder_path = glob(video_glob)
    # pass_folder_path = glob(os.path.join(root, "Senator_ig_videos_preprocess/*mp4"))

    # filtered_all_folder = (removeExistFile(folder_path, pass_folder_path))
    filtered_all_folder = folder_path
    
    # Determine the number of available CPU cores
    # num_workers = min(cpu_count(), len(folder_path)) //3
    num_workers = 4
 
    print(f"Starting processing with {num_workers} workers using CUDA...")
    # exit()
    
    # Process videos in parallel
    process_videos_in_parallel(filtered_all_folder, num_workers, save_root=root , save_folder= save_folder)
