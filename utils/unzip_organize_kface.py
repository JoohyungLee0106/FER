import zipfile
import glob
import os
import shutil
import multiprocessing
from itertools import repeat

# 영상 확장자
EXT='.jpg'

# 본 코드의 input: zip파일 저장 위치
ZIP_PATHS = '/media/keti/8tb/kface_zip'

# 본 코드의 output: 본 코드를 통하여 압축이 풀리고 저장될 위치
ZIP_PATHS_NEW = '/media/keti/8tb/kface'
# PERFIX_FILENAME = 'EMOIMG_'

# 데이터셋 split
SPLITS = ['train', 'val', 'test']

# 감정 클래스
CATEGORIES = ['Neutral', 'Happy', 'Sad']

for split in SPLITS:
    if os.path.isdir(os.path.join(ZIP_PATHS_NEW, split)):
        shutil.rmtree(os.path.join(ZIP_PATHS_NEW, split))
    os.mkdir(os.path.join(ZIP_PATHS_NEW, split))
    for category in CATEGORIES:
        if os.path.isdir(os.path.join(ZIP_PATHS_NEW, split, category)):
            shutil.rmtree(os.path.join(ZIP_PATHS_NEW, split, category))
        os.mkdir(os.path.join(ZIP_PATHS_NEW, split, category))

def extract_one_zip(path_zip, split):
    zipdata = zipfile.ZipFile(path_zip)
    zipinfos = zipdata.infolist()

    name_zip = os.path.basename(path_zip).replace('.zip','')

    for i, zipinfo in enumerate(zipinfos):
        # This will do the renaming
        if zipinfo.filename[-3:] == 'jpg':
            emotion_cat = zipinfo.filename.split('/')[2]
            emotion_num = int(emotion_cat[1:]) - 1
            zipinfo.filename = zipinfo.filename.replace(emotion_cat + '/', '')
            zipinfo.filename = os.path.join(ZIP_PATHS_NEW, split, CATEGORIES[emotion_num], name_zip+'_'+zipinfo.filename.replace('/', '_'))
            zipdata.extract(zipinfo)


list_all = glob.glob(os.path.join(ZIP_PATHS, '*.zip'))

list_train = list_all[:round(len(list_all)*0.8)]
list_val = list_all[round(len(list_all)*0.8):round(len(list_all)*0.9)]
list_test = list_all[round(len(list_all)*0.9):]

list_split = [list_train, list_val, list_test]

for split_name, split in zip(SPLITS, list_split):
    with multiprocessing.Pool(processes=16) as pool:
        pool.starmap(extract_one_zip, zip(split, repeat(split_name)))
#
# pool = ThreadPool(processes=8)
# pool.map(extract_one_zip, (path_zip for path_zip in list_train))
# pool.close()
# pool.join()
#
# pool = ThreadPool(processes=8)
# pool.map(extract_one_zip, (path_zip for path_zip in list_val))
# pool.close()
# pool.join()
#
# pool = ThreadPool(processes=8)
# pool.map(extract_one_zip, (path_zip for path_zip in list_test))
# pool.close()
# pool.join()


#
# p = Pool(processes=8)
# p.map(extract_one_zip, list_train)
# p.close()
# p.join()

