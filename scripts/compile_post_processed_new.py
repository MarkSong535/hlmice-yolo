import numpy as np
import glob
import heapq
import re
import platform

if platform.system() == "Darwin":
    DEBUG = True
else:
    DEBUG = False

"""Indeices: {0:'headstage',1:'snout',2:'body',3:'ball',4:'pellet',5:'brown_nest',6:'white_nest',7:'box',8:'food_gel',9:'spout',10:'porthole'}"""

output_dir = "./yolov5/runs/detect/out/labels/"
dlc_path = "/models/DLC/"

if DEBUG:
    output_dir = "./yolov5/runs/detect/out/labels/"
    dlc_path = "data_converter/data_organized/aj_ris/"

row_structure = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
available_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

closest_organize = [4,10]

def distance(p1, p2):
    _p1 = np.array(p1, dtype=np.float32)
    _p2 = np.array(p2, dtype=np.float32)
    return np.sqrt(np.sum((_p1 - _p2) ** 2))

files = glob.glob(output_dir + "*")
get_index = lambda filename: re.search(r"\d_(\d+).txt", filename).groups()[0]
get_filename = lambda filename: re.search(
    r"(\w+_\w+-\w+-\w+)_\w+.txt", filename
).groups()[0]

videos = set()
for file in files:
    videos.add(get_filename(file))

for video in videos:
    files_video = [file for file in files if video in file]
    sorted_files = []
    for file in files_video:
        score = int(get_index(file))
        heapq.heappush(sorted_files, (score, file))
        
    DLC_file = np.load(dlc_path+video+".npy")[:,0,:].reshape(-1,2)

    data = np.full(
            (DLC_file.shape[0] ,(np.max(available_indices) + 1) * 5), -np.inf
        ) 
    counter = 0
    while sorted_files != []:
        idx, file = sorted_files.pop(0)
        with open(file, "r") as f:
            line = f.read()
        entries = line.split("\n")[:-1]

        placeholder = np.full(
            ((np.max(available_indices) + 1) * 5), -np.inf
        )  # 4 coordinated for bounding boxes and 1 for confidence

        # iterate through entries
        detected = []
        for object_data in entries:
            points = object_data.split(" ")
            id = int(points[0])
            start_idx = np.where(np.array(row_structure) == id)[0]

            if id not in available_indices:
                print(f"Found {id} but not in available indices")
                continue
            
            if not id in closest_organize:
                conf_new = float(points[-1])
                for idx in start_idx:
                    conf = placeholder[idx * 5 + 4]
                    if len(start_idx) > 1 and conf > -np.inf:
                        #print(f"{conf_new} {conf}")
                        continue
                    if conf_new > conf:
                        for i in range(len(points) - 1):
                            placeholder[idx * 5 + i] = float(points[i + 1])
                        detected.append(id)
                        break
            else:
                if not placeholder[id * 5] == -np.inf:
                    if counter >= DLC_file.shape[0]:
                        continue
                    dist_new = distance(DLC_file[counter], points[1:3])
                    dist_old = distance(DLC_file[counter], placeholder[id * 5 : id * 5 + 2])
                    if dist_new < dist_old:
                        for i in range(len(points) - 1):
                            placeholder[id * 5 + i] = float(points[i + 1])
                        detected.append(id)
                else:
                    for i in range(len(points) - 1):
                        placeholder[id * 5 + i] = float(points[i + 1])
                    detected.append(id)

        # Special Handeling for the balls
        # ball_indices = np.where(np.array(row_structure) == 3)[0]
        # if not ball_indices.shape[0] < 2:
        #     if placeholder[ball_indices[0] * 5] != -np.inf and placeholder[ball_indices[1] * 5] != -np.inf:
        #         if placeholder[ball_indices[0] * 5 + 1] >= placeholder[ball_indices[1] * 5 + 1]:
        #             temp = placeholder[ball_indices[0] * 5 : ball_indices[0] * 5 + 5].copy()
        #             placeholder[ball_indices[0] * 5 : ball_indices[0] * 5 + 5] = placeholder[ball_indices[1] * 5 : ball_indices[1] * 5 + 5]
        #             placeholder[ball_indices[1] * 5 : ball_indices[1] * 5 + 5] = temp

        if set(detected) != set(row_structure):
            set_not_found = set(row_structure) - set(detected)
            print(f"The follow objects are not detected: {set_not_found}")

        placeholder[placeholder == -1] = np.nan
        data[idx] = placeholder
    np.save(f"./{video}_processed_array.npy", data)
