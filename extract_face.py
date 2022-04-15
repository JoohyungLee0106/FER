import os
import shutil
# This code uses MTCNN from   https://github.com/timesler/facenet-pytorch
# NUM_IMAGES: 1) positive: from the first, 2) negative: from the last, 3) 0: all images

# 본 코드의 input: 클래스 별 영상이 저장되어 있는 폴더
DIR_IN = 'data/raw_images_eng'

# 본 코드의 output: 클래스 별 얼굴 영상이 저장될 폴더
DIR_OUT = 'data/train'
# DIR_OUT 폴더 아래의 각 클래스별 폴더에서 변환할 영상 갯수. 0이면 전부 변환합니다.
NUM_IMAGES = 0

# DIR_OUT = 'data/test'
# NUM_IMAGES = 0

if __name__ == "__main__":
    # DIR_OUT 의 폴더를 만듭니다.
    if os.path.isdir(DIR_OUT):
        shutil.rmtree(DIR_OUT)
    os.mkdir(DIR_OUT)

    # utils/extract_face.py 를 args.input, args.output, args.num_images를 변환하여 실행시킵니다.
    for emotion in os.listdir(DIR_IN):
        os.system(f"python utils/extract_face.py --input {DIR_IN}/{emotion} --output {DIR_OUT}/{emotion} --num-images {NUM_IMAGES}")
