import numpy as np
import glob
import heapq
import re
import platform

if platform.system() == "Darwin":
    DEBUG = True
else:
    DEBUG = False

output_dir = "./yolov5/runs/detect/out/labels/"
# dlc_path = "/models/DLC/"

if DEBUG:
    output_dir = "./yolov5/runs/detect/out/labels/"
    dlc_path = "data_converter/data_organized/aj_ris/"

row_structure = [0]
available_indices = [0]

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

    data = np.full(
            (100000 ,(len(row_structure)) * 5), -np.inf
        ) 
    r_data_list = ['nan' for _ in range(100000)]
    while sorted_files != []:
        midx, file = sorted_files.pop(0)
        with open(file, "r") as f:
            line = f.read()
        entries = line.split("\n")[:-1]

        placeholder = np.full(
            ((len(row_structure)) * 5), -np.inf
        )  # 4 coordinated for bounding boxes and 1 for confidence

        # iterate through entries
        for object_data in entries:
            r_data_list[midx-1] = r_data_list[midx-1]+"next:"+object_data
            points = object_data.split(" ")
            id = int(points[0])
            start_idx = np.where(np.array(row_structure) == id)[0]

            conf_new = float(points[-1])
            for idx in start_idx:
                conf = placeholder[idx * 5 + 4]
                if len(start_idx) > 1 and conf > -np.inf:
                    #print(f"{conf_new} {conf}")
                    continue
                if conf_new > conf:
                    for i in range(len(points) - 1):
                        placeholder[idx * 5 + i] = float(points[i + 1])
                    break

        placeholder[placeholder == -1] = np.nan
        data[midx-1] = placeholder
    np.save(f"./{video}_processed_array.npy", data)
    with open(f"./{video}_processed_raw.npy", "w") as f:
        for line in r_data_list:
            f.write(line + "\n")
