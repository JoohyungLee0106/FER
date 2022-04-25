import argparse
import torch
import os
# import torchvision.models as models
# import torchvision.transforms as transforms
import numpy as np
import model as models
from PIL import Image
import torchvision.datasets as datasets
from utils.transforms import transforms_test as transform
from utils.transforms import NormalizePerImage
import glob
from PIL import Image, ImageFont, ImageDraw
import shutil
from utils.facenet_pytorch import MTCNN
import cv2

parser = argparse.ArgumentParser(description='Emotion inference from cropped face image')
parser.add_argument('--model', type=str, default='checkpoints/efficientNetV2_m.pth.tar')

# AI-HUB: ['HAPPY', 'EMBARRASED', 'ANGRY', 'ANXIOUS', 'HURT', 'SAD', 'NEUTRAL']
# kface: ['HAPPY', 'NEUTRAL', 'SAD']
parser.add_argument('--emotions', default=['HAPPY', 'EMBARRASED', 'ANGRY', 'ANXIOUS', 'HURT', 'SAD', 'NEUTRAL'], type=str, help='emotion categories')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--resize-h', type=float, default=0)
parser.add_argument('--fps', type=int, default=50)
parser.add_argument('--entropy-threshold', type=float, default=2)

args = parser.parse_args()

# EMOTION = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
# EMOTION =['HAPPY', 'EMBARRASED', 'ANGRY', 'ANXIOUS', 'HURT', 'SAD', 'NEUTRAL']
FONT_SCALE=0.5
FONT_COLOR=(255,255,255)
FONT_THICKNESS=2
normalize=NormalizePerImage()
torch.cuda.set_device(args.gpu)

camera = cv2.VideoCapture('rtsp://192.168.10.101:554/media/1/1')
# camera = cv2.VideoCapture(0)

mtcnn = MTCNN(image_size=160, margin=8, device=torch.device(f'cuda:{args.gpu}'))
# mtcnn = MTCNN(image_size=160, margin=8)


if 'effnetv2_s' in os.path.basename(args.model):
    # import model.efficientNetV2 as models
    model = models.__dict__['effnetv2_s'](num_classes=len(args.emotions))
elif 'effnetv2_m' in os.path.basename(args.model):
    # import model.efficientNetV2 as models
    model = models.__dict__['effnetv2_m'](num_classes=len(args.emotions))
elif 'effnetv2_l' in os.path.basename(args.model):
    # import model.efficientNetV2 as models
    model = models.__dict__['effnetv2_l'](num_classes=len(args.emotions))
elif 'mobilenet_v3_small' in os.path.basename(args.model):
    # import torchvision.models as models
    model = models.__dict__['mobilenet_v3_small'](num_classes=len(args.emotions))
elif 'mobilenet_v3_large' in os.path.basename(args.model):
    # import torchvision.models as models
    model = models.__dict__['mobilenet_v3_large'](num_classes=len(args.emotions))
else:
    raise ValueError('Invalid model !!!')

# 명시된 gpu로의 모델 카피
model = model.cuda(args.gpu)
# args.model 에서 모델weight 을 불러와서 args.gpu에 카피.
checkpoint = torch.load(args.model, map_location=f'cuda:{args.gpu}')
# 위에서 카피된 모델weight를 gpu로 카피된 모델에 로딩시킴
model.load_state_dict(checkpoint['state_dict'])
# 훈련이 아니므로 eval 모드로 전환
model.eval()

cv2.namedWindow('DEMO')

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if (int(major_ver) < 3):
    FPS = camera.get(cv2.CV_CAP_PROP_FPS)
else:
    FPS = camera.get(cv2.CAP_PROP_FPS)

# assert(args.fps < FPS)
# SKIP_FRAME = round(FPS/float(args.fps))

i = 0
while True:
    i += 1
    # if i == SKIP_FRAME:
    check, frame = camera.read()
    print(f'frame: {frame.shape}')
    if args.resize_h > 0:
        frame = cv2.resize(frame, (int(args.resize_h), int(args.resize_h*frame.shape[0]/float(frame.shape[1]))))

    # draw = ImageDraw.Draw(Image.fromarray(frame))
    if check:

        # with torch.cuda.amp.autocast():
        faces, boxes = mtcnn(frame)
        if faces != None:
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    face = normalize(faces).cuda(args.gpu, non_blocking=True)
                    # face = normalize(faces)
                    emotion = model(torch.unsqueeze(face, 0))
                    prob = torch.nn.functional.softmax(emotion)
                    entropy = -torch.sum(prob * torch.log(prob))
                    # print(f'entropy: {entropy}')
                cv2.rectangle(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (255, 0, 0), 2)
                if entropy < args.entropy_threshold:
                    cv2.putText(frame, args.emotions[torch.argmax(emotion)], (int(boxes[0][0]), int(boxes[0][1])-10), 0, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

                # draw.rectangle(boxes[0].tolist(), outline=(255, 0, 0), width=6)
                # draw.text((int(boxes[0][0]), int(boxes[0][3])+3), args.emotions[torch.argmax(emotion)], font=ImageFont.truetype("fonts/gulim.ttc", 20), align ="left")
                    print(f'emotion: {args.emotions[torch.argmax(emotion)]}')

        # 얼굴위치박스와 감정 텍스트가 덧입혀진 frame을 화면에 띄워줍니다
        cv2.imshow('DEMO', frame)

        # args.fps기준으로 기다립니다.
        cv2.waitKey(int(1000/float(args.fps)))

    # q 를 누르면 꺼집니다. 안꺼지면 ctrl+c 누르면 됩니다.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

