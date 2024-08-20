import os
import sys

IS_MULTIANIMAL = True 
PATH_INCLUDES_MODEL = True

video_root = "/media/storage/debug_videos/"
video_name = None
if video_root == None:
    print('Video root path not specified in system environment (in .sh file)')
    raise ValueError('Video root path not specified in system environment (in .sh file)')
else:
    print("analyse video in "+video_root)

print(os.system('nvidia-smi'), flush=True)
for k, v in os.environ.items():
    print(f'{k}={v}')
gpu_num = os.getenv('CUDA_VISIBLE_DEVICES')
print("gpu_num ", gpu_num)
print("type gpu_num ", type(gpu_num), flush=True)
print('os.environ["CUDA_VISIBLE_DEVICES"] = ', str(int(gpu_num)), flush=True)
# os.environ["CUDA_VISIBLE_DEVICES"] = str(int(gpu_num))
# for k, v in os.environ.items():
#     print(f'{k}={v}')
# gpu_num2 = os.getenv('CUDA_VISIBLE_DEVICES')
# print("gpu_num2 ", gpu_num2, flush=True)
print(os.system('nvidia-smi'), flush=True)

#import tensorflow as tf
## gpus = tf.config.experimental.list_physical_devices('GPU')
## if gpus:
#if 1:
#  # Restrict TensorFlow to only use the first GPU
#  try:
#    # tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#    tf.config.experimental.set_visible_devices(gpus[int(gpu_num)], 'GPU')
#  except RuntimeError as e:
#    # Visible devices must be set at program startup
#    print(e)

import tensorflow as tf

tf.__version__
print(tf.test.gpu_device_name())
tf.config.list_physical_devices('GPU')

import os
os.environ["DLClight"]="True"
#os.environ["DLClight"]="False"
os.environ["HDF5_USE_FILE_LOCKING"]="False"
# now we are ready to train!
import deeplabcut
deeplabcut.__version__

import glob
import numpy as np
import os.path as op

# check gpu
print(os.system('nvidia-smi'))
if tf.test.is_gpu_available():
    print("GPU")
else:
    raise RuntimeError("No GPU available")

os.chdir('/media/storage/')

# change to your path
if PATH_INCLUDES_MODEL:
    path_config_file = '/media/storage/model/config.yaml'
else:
    path_config_file = '/media/storage/config.yaml'
# check path_config_file exists
if os.path.exists(path_config_file) and os.path.isfile(path_config_file):
    print("path_config_file ", path_config_file)
else:
    raise NotADirectoryError("Directory {} does not exists".format(path_config_file))

# deeplabcut.check_labels(path_config_file,draw_skeleton=False )
# Create a training dataset
#deeplabcut.create_training_dataset(path_config_file,Shuffles=[1], windows2linux=True)
#
# Start training
#reset in case you started a session before...
# tf.reset_default_graph()
#deeplabcut.train_network(path_config_file, shuffle=1)
#this will run until you stop it (CTRL+C), or hit "STOP" icon,
#or when it hits the end (default, 1.3M iterations).


#videofile_path = ['/media/storage/CAF00072-20210307T135414-145415.mp4']

if not video_root[-1] == "/":
    video_root += "/"

print(video_root+'*.mp4')

if not video_name == None:
    videofile_path = [video_root+video_name+'.mp4']
else:
    videofile_path = np.sort(glob.glob(video_root+'*.mp4'))

print(videofile_path)

print("video going to process:")

for i in videofile_path:
    print("\t"+i)

# before procedeeing check for labeled videos
#for vfl in videofile_path:
 #   print(vfl)
  #  if "labeled" in vfl:
   #     raise ValueError("videofile_path contains labeled video, {}"
    #                     .format(vfl))

deeplabcut.evaluate_network(path_config_file,plotting=True)

deeplabcut.analyze_videos(path_config_file,videofile_path)
print("Finished analyzing videos")

deeplabcut.create_labeled_video(path_config_file,videofile_path)
print("Finished labelling videos")

if IS_MULTIANIMAL:
    deeplabcut.create_video_with_all_detections(path_config_file,videofile_path)
    print("Finished labelling videos for all")

# deeplabcut.plot_trajectories(path_config_file,videofile_path,showfigures=True)