import cv2
from multiprocessing.pool import ThreadPool
import glob
import os
from facenet_pytorch import MTCNN
import argparse
import torch
import shutil
import random

parser = argparse.ArgumentParser(description='Extract face portion and save')
parser.add_argument('--input', default='../data/raw_images_eng/ANGRY', help='directory where raw images are saved')
parser.add_argument('--output', default='../data/face_images_eng/ANGRY', help='directory where cropped images will be saved')
parser.add_argument('--file-type', default='jpg', type=str, help='file type, e.g., jpg')
parser.add_argument('--num-process', default=16, type=int, help='number of process')
parser.add_argument('--image-size', default=160, type=int, help='size of extracted face image')
parser.add_argument('--face-margin', default=15, type=int, help='additional spatial margin around face')
parser.add_argument('--hierarchy-level', default=1, type=int, help='hierarchy level of files saved')
parser.add_argument('--gpu', default=0, type=int, help='hierarchy level of files saved')
parser.add_argument('--num-images', default=0, type=int, help='number of images to use per emotion category. Max: 7000')

def main():
    args = parser.parse_args()

    if os.path.isdir(args.output):
        shutil.rmtree(args.output)

    os.mkdir(args.output)

    mtcnn = MTCNN(image_size=args.image_size, margin=args.face_margin, device=torch.device(f'cuda:{args.gpu}'))

    def extract_and_save(path, args=args):
        img = cv2.imread(path)

        # 중요!!
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            torch.cuda.empty_cache()
            _ = mtcnn(img, path.replace(args.input, args.output))
        except Exception as e:
            print(f'{os.path.basename(path)} : {e}')

    pp = glob.glob(args.input + ('/*' * args.hierarchy_level) + f'.{args.file_type}')
    #pp.sort()
    random.seed(3)
    random.shuffle(pp)

    if args.num_images > 0:
        pp = pp[:args.num_images]
    elif args.num_images < 0:
        pp = pp[args.num_images::]

    if args.num_process > 2:
        #pass
        pool = ThreadPool(processes=args.num_process)
        pool.map(extract_and_save, (img_path for img_path in pp))
        pool.close()
        pool.join()
    else:
        for img_path in pp:
            extract_and_save(img_path, args)

if __name__ == '__main__':
    main()
