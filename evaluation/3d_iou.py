import sys
import os
import subprocess

python_exe = 'C:\\Program Files\\Blender Foundation\\Blender 3.2\\3.2\\python\\bin\\python.exe'
print(python_exe)
subprocess.call([python_exe, "-m", "ensurepip"])
subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])

subprocess.call([python_exe, "-m", "pip", "install", "numpy"])
subprocess.call([python_exe, "-m", "pip", "install", "opencv-python"])

subprocess.call([python_exe, "-u render_batch --model_root_dir {model root dir} --render_root_dir {where you store images} --filelist_dir {which models you want to render} --blender_location {you} --num_thread {10} --shapenetversion {support v1, v2} --debug {False}"])