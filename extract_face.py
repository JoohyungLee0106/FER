import sys
import os
import shutil
import argparse

DIR_IN = 'data/raw_images_eng'
DIR_OUT = 'data/train'

if __name__ == "__main__":
    if os.path.isdir(DIR_OUT):
        shutil.rmtree(DIR_OUT)
    os.mkdir(DIR_OUT)

    for emotion in os.listdir(DIR_IN):
        os.system(f"python utils/extract_face.py --input {DIR_IN}/{emotion} --output {DIR_OUT}/{emotion} --num-images 10")