"""
Activate your virtual environment before running, e.g.:
source /path/to/your/venv/bin/activate
"""
import cv2
import mediapipe as mp
import face_recognition
import numpy as np
import csv
import os
from tqdm import tqdm
import datetime
import pandas as pd


def initialize_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Error: Could not open video: {video_path}")
    frame_rate = int(video_capture.get(cv2.CAP_PROP_FPS)) # get video fps
    return video_capture, frame_rate

def detect_faces_mediapipe(frame, face_detection):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    face_locations = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            (x, y, w, h) = bbox
            face_locations.append((y, x + w, y + h, x))

    return face_locations

def recognize_faces(rgb_frame, face_locations, known_face_encodings, known_face_names):
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, model="cnn")
    face_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

    return face_names

def draw_results(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)


def save_cropped_faces(frame, face_locations, face_names, save_png, frame_count, second, person):
    saved_filenames = [] 
    padding = 5 

    for idx, ((top, right, bottom, left), name) in enumerate(zip(face_locations, face_names)):
        top = max(0, top - padding)
        bottom = min(frame.shape[0], bottom + padding)
        left = max(0, left - padding)
        right = min(frame.shape[1], right + padding)

        cropped_face = frame[top:bottom, left:right]
        face_filename = f"{save_png}_{second}_frame_{frame_count}_face_{person}.png"

        cv2.imwrite(face_filename, cropped_face)
        print(f"Saved cropped face {idx} from frame {frame_count} to {face_filename}")

        saved_filenames.append(face_filename) 

    return saved_filenames  



def log_results_to_csv(csv_writer, second, frame_count, face_count, recognized_names, target_names, face_filename, sameFlag):
    is_target_present = any(name in target_names for name in recognized_names)
    csv_writer.writerow({
        "Second": second,
        "Time": str(datetime.timedelta(seconds=second)),
        "Frame count": frame_count,
        "Face Count": face_count,
        "Target Present": int(is_target_present),
        "Same Person": sameFlag,
        "Recognized Names": ", ".join(recognized_names) if recognized_names else "None",
        "Cropped File": face_filename if face_filename else "None"
    })


def process_video(video_path, known_face_encodings, known_face_names, save_csv, target_name):
    video_capture, frame_rate = initialize_video(video_path)
    frame_count = 0
    #frame_interval = max(1, frame_rate // 3) # 代表每秒分析 3 幀
    frame_interval = 10
    os.makedirs(os.path.dirname(save_csv), exist_ok=True)

    with open(save_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        fieldnames = ["Second", "Time", "Frame count", "Face Count", "Target Present", "Same Person", "Recognized Names", "Cropped File"]
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()

        mp_face_detection = mp.solutions.face_detection
        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
            try:
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break

                    frame_count += 1
                    if frame_count % frame_interval != 0:
                        continue

                    second = frame_count // frame_rate
                    face_locations = detect_faces_mediapipe(frame, face_detection)

                    face_filenames = []
                    samePerson = 0


                    if face_locations:
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        face_names = recognize_faces(rgb_frame, face_locations, known_face_encodings, known_face_names)

                        save_png = save_csv.replace(".csv", "")

                        """
                        # 只找影片主人
                        if target_name in face_names:
                            filtered_locations = [loc for loc, name in zip(face_locations, face_names) if name == target_name]
                            face_filenames = save_cropped_faces(frame, filtered_locations, [target_name], save_png, frame_count, second)
                        """
                        # 每個人獨立存檔
                        for person, location in zip(face_names, face_locations):
                            if person != "Unknown":
                                saved_files = save_cropped_faces(frame, [location], [person], save_png, frame_count, second, person)
                                face_filenames.extend(saved_files)  
                            
                                if person == target_name:  
                                    samePerson = 1

                        face_filename = ", ".join(face_filenames) if face_filenames else "None"
                        log_results_to_csv(csv_writer, second, frame_count, len(face_names), face_names, known_face_names, face_filename, samePerson)

                        print(f"Second {second}: {len(face_names)} people detected, names: {face_names}")

                    else:
                        log_results_to_csv(csv_writer, second, frame_count, 0, [], known_face_names, "None", samePerson)
                        print(f"Second {second}: No faces detected.")

                    cv2.imshow('Video Processing', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            finally:
                video_capture.release()
                cv2.destroyAllWindows()

def load_known_faces(image_paths, names):
    known_face_encodings = []
    for img_path in tqdm(image_paths):
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_face_encodings.append(encoding)
    return known_face_encodings

def load_photo_label(image_paths):
    known_images = []
    known_face_names = []
    for root, dirs, files in os.walk(image_paths):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                known_images.append(os.path.join(root, file))
                known_face_names.append(root.split(os.sep)[-1].replace("_", " "))

    if len(known_images) != len(known_face_names):
        raise ValueError("Error: label != number!")
    return known_images, known_face_names

def loop_file(mp4_paths):
    print("!!!start loop file!!!!")
    input_paths = []
    save_paths = []

    for root, dirs, files in os.walk(mp4_paths):
        for file in files:
            if file.endswith(".mp4"):
                input_paths.append(os.path.join(root, file))
                name = root.split(os.sep)[-1]
                if name.split("_")[-1] == "primary":
                    name = name.split("_")[0] + "_" + name.split("_")[1]
                save_paths.append(os.path.join(name, file))
    return input_paths, save_paths

def loop_dataframe(df, root, save_root):
    input_paths = []
    save_paths = []

    for row in df.dropna().itertuples(index=False):
        file = str(row[0])  

        input_file = os.path.join(root, file)
        save_file = os.path.join(save_root, file)

        input_paths.append(input_file)
        save_paths.append(save_file)

    return input_paths, save_paths


if __name__ == "__main__":
    # Set these via environment variables, or edit the defaults below.
    mp4_root = os.environ.get("MP4_ROOT", "./data/ig_videos_raw/")
    image_root = os.environ.get("IMAGE_ROOT", "./data/senator_image/")
    save_root = os.environ.get("SAVE_ROOT", "./output/senator_recognition/")

    reference_file = os.environ.get(
        "REFERENCE_FILE",
        "./data/ig_metadata_processed_with_zscore.csv",
    )
    _data = pd.read_csv(reference_file)
    _valid_file = _data[["video_root"]]
    input_paths, save_paths = loop_dataframe(_valid_file, mp4_root,save_root)

    # input_paths, save_paths = loop_file(mp4_paths=mp4_root)
    known_images, known_face_names = load_photo_label(image_root)
    known_face_encodings = load_known_faces(known_images, known_face_names)

    for idx, video_path in tqdm(enumerate(input_paths[:]), desc="Processing videos", total=len(input_paths)):

        file_prefix= os.path.basename(save_paths[idx]).replace(".mp4", "")
        name = file_prefix[:-12].replace("_", " ")
        save_csv =os.path.join(save_root,name, file_prefix, file_prefix+".csv")

        # save_csv = save_csv.replace(".mp4", ".csv")

        print("name",name)
        print("save_csv",save_csv)

        # exit()
        os.makedirs(os.path.dirname(save_csv), exist_ok=True)
        process_video(video_path, known_face_encodings, known_face_names, save_csv, name)

        print(f"Results saved to: {save_csv}")
