import cv2
from multiprocessing.pool import ThreadPool
import glob
import os
import torch
import shutil as sh
from facenet_pytorch import MTCNN, InceptionResnetV1

path_list1 = ['/home/hyj/SSD8/emotion/Training/EMOIMG_중립_TRAIN_02/','/home/hyj/SSD8/emotion/Training/EMOIMG_중립_TRAIN_03/','/home/hyj/SSD8/emotion/Training/EMOIMG_중립_TRAIN_04/']
path_list2 = ['/home/hyj/SSD8/emotion/Training/EMOIMG_기쁨_TRAIN_02/','/home/hyj/SSD8/emotion/Training/EMOIMG_기쁨_TRAIN_03/','/home/hyj/SSD8/emotion/Training/EMOIMG_기쁨_TRAIN_04/']
path_list3 = ['/home/hyj/SSD8/emotion/Training/EMOIMG_슬픔_TRAIN_02/','/home/hyj/SSD8/emotion/Training/EMOIMG_슬픔_TRAIN_03/','/home/hyj/SSD8/emotion/Training/EMOIMG_슬픔_TRAIN_04/']
path_list4 = ['/home/hyj/SSD8/emotion/Training/EMOIMG_불안_TRAIN_02/','/home/hyj/SSD8/emotion/Training/EMOIMG_불안_TRAIN_03/','/home/hyj/SSD8/emotion/Training/EMOIMG_불안_TRAIN_04/']
path_list5 = ['/home/hyj/SSD8/emotion/Training/EMOIMG_상처_TRAIN_02/','/home/hyj/SSD8/emotion/Training/EMOIMG_상처_TRAIN_03/','/home/hyj/SSD8/emotion/Training/EMOIMG_상처_TRAIN_04/']
path_list6 = ['/home/hyj/SSD8/emotion/Training/EMOIMG_분노_TRAIN_02/','/home/hyj/SSD8/emotion/Training/EMOIMG_분노_TRAIN_03/','/home/hyj/SSD8/emotion/Training/EMOIMG_분노_TRAIN_04/']
path_list7 = ['/home/hyj/SSD8/emotion/Training/EMOIMG_당황_TRAIN_02/','/home/hyj/SSD8/emotion/Training/EMOIMG_당황_TRAIN_03/','/home/hyj/SSD8/emotion/Training/EMOIMG_당황_TRAIN_04/']

def file_move(move_path):
    for path in path_list7:
        file_list = glob.glob(os.path.join(path,'*'))
        for file in file_list:
            sh.move(file ,move_path)

def rename_file():
    i = 0
    j = 0
    for path in path_list1:
        file_list = glob.glob(os.path.join(path,'*'))
        for file in file_list:
            os.rename(file , os.path.split(file)[0] + '/' +str(j) + '_'+ str(i) + '.jpg')
            i+=1
        i = 0
        j += 1

#file_move('/home/hyj/SSD8/emotion/Training/EMOIMG_당황_TRAIN_01/')
#rename_file()
def facecrop(img , path):
    facedata = "haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(facedata)

    minisize = (img.shape[1], img.shape[0])
    miniframe = cv2.resize(img, minisize)

    faces = cascade.detectMultiScale(miniframe)

    for f in faces:
        x, y, w, h = [v for v in f]
        face = img[y:y + h, x:x + w]
        face = cv2.resize(face, dsize=(160, 160))
        os.makedirs(path)
        cv2.imwrite(path, face)
        cv2.waitKey(0)

    return

PATH = '/home/hyj/SSD8/emotion/Training/'
mtcnn = MTCNN(image_size=160, margin=15)

def extract_and_save(path, ratio=False):
    path_new = path.replace('emotion/', 'emotion_mtcnn_built_in/')
    img = cv2.imread(path)
    if img is not None:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
        _ = mtcnn(img, path_new)

# pool = ThreadPool(processes=12)
# pool.map(extract_and_save, (img_path for img_path in glob.glob(os.path.join(PATH,'*','*'))) )

for img in glob.glob(os.path.join(PATH,'0','*')):
    extract_and_save(img)
