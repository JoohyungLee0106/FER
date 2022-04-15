import zipfile
import glob
import os
from multiprocessing import Pool

# 영상 확장자
EXT='.jpg'

# 본 코드의 input: zip파일 저장 위치
ZIP_PATHS = '/media/keti/8tb/aihub_fer/zip/Training'

# 본 코드의 output: 본 코드를 통하여 압축이 풀리고 저장될 위치. 폴더를 미리 만들어둬야 함.
ZIP_PATHS_NEW = '../data/aihub/train'
# PERFIX_FILENAME = 'EMOIMG_'

# 감정 클래스
CATEGORIES = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']

for category in CATEGORIES:
    if not os.path.isdir(os.path.join(ZIP_PATHS_NEW, category)):
        os.mkdir(os.path.join(ZIP_PATHS_NEW, category))

def extract_one_zip(path_zip):
    zipdata = zipfile.ZipFile(path_zip)
    zipinfos = zipdata.infolist()

    if os.path.basename(path_zip)[7:9] in CATEGORIES:
        category_num = CATEGORIES.index(os.path.basename(path_zip)[7:9])
        category_str = CATEGORIES[category_num]

        # iterate through each file
        for i, zipinfo in enumerate(zipinfos):
            # This will do the renaming
            # zipinfo.filename = os.path.join(ZIP_PATHS_NEW, category_str, str(i) + EXT)
            zipinfo.filename = os.path.join(ZIP_PATHS_NEW, category_str, f'{os.path.basename(path_zip).replace(".zip","")[-2:]}_{i}{EXT}')
            zipdata.extract(zipinfo)

p = Pool(processes=7)
p.map(extract_one_zip, glob.glob(os.path.join(ZIP_PATHS, '*.zip')))
p.close()
p.join()



# for path_zip in glob.glob(os.path.join(ZIP_PATHS, '*.zip')):
#     zipdata = zipfile.ZipFile(path_zip)
#     zipinfos = zipdata.infolist()
#
#     if os.path.basename(path_zip)[7:9] in CATEGORIES:
#         category_num = CATEGORIES.index(os.path.basename(path_zip)[7:9])
#         category_str = CATEGORIES[category_num]
#
#         # iterate through each file
#         for i, zipinfo in enumerate(zipinfos):
#             # This will do the renaming
#             zipinfo.filename = os.path.join(ZIP_PATHS_NEW, category_str, f'{os.path.basename(path_zip).replace(".zip","")[-2:]}_{i}{EXT}')
#             zipdata.extract(zipinfo)
