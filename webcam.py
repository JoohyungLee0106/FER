import argparse
import torch
import os
import torchvision.models as models
# import torchvision.transforms as transforms
import numpy as np
import model.efficientNetV2 as models
from PIL import Image
import torchvision.datasets as datasets
from utils.transforms import transformation_val as transform
from utils.transforms import NormalizePerImage
import glob
from PIL import Image, ImageFont, ImageDraw
import shutil
from utils.facenet_pytorch import MTCNN
import cv2

parser = argparse.ArgumentParser(description='Emotion inference from cropped face image')
parser.add_argument('--model', type=str, default='checkpoints/efficientNetV2_m.pth.tar')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--resize-h', type=float, default=0)
parser.add_argument('--fps', type=int, default=50)
parser.add_argument('--entropy-threshold', type=float, default=2)

args = parser.parse_args()

# EMOTION = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
EMOTION =['HAPPY', 'EMBARRASED', 'ANGRY', 'ANXIOUS', 'HURT', 'SAD', 'NEUTRAL']
FONT_SCALE=0.5
FONT_COLOR=(255,255,255)
FONT_THICKNESS=2

camera = cv2.VideoCapture('rtsp://192.168.10.101:554/media/1/1')
# camera = cv2.VideoCapture(0)

mtcnn = MTCNN(image_size=160, margin=8, device=torch.device('cuda:0'))
# mtcnn = MTCNN(image_size=160, margin=8)

efficient_net = models.__dict__['effnetv2_m'](num_classes=7)
normalize=NormalizePerImage()

torch.cuda.set_device(0)
efficient_net = efficient_net.cuda(0)
checkpoint = torch.load(args.model, map_location='cuda:0')
# checkpoint = torch.load(args.model)

efficient_net.load_state_dict(checkpoint['state_dict'])
efficient_net.eval()

cv2.namedWindow('DEMO')

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

if (int(major_ver) < 3):
    FPS = camera.get(cv2.CV_CAP_PROP_FPS)
else:
    FPS = camera.get(cv2.CAP_PROP_FPS)

# assert(args.fps < FPS)
SKIP_FRAME = round(FPS/float(args.fps))

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
                    emotion = efficient_net(torch.unsqueeze(face, 0))
                    prob = torch.nn.functional.softmax(emotion)
                    entropy = -torch.sum(prob * torch.log(prob))
                    # print(f'entropy: {entropy}')
                cv2.rectangle(frame, (int(boxes[0][0]), int(boxes[0][1])), (int(boxes[0][2]), int(boxes[0][3])), (255, 0, 0), 2)
                if entropy < args.entropy_threshold:
                    cv2.putText(frame, EMOTION[torch.argmax(emotion)], (int(boxes[0][0]), int(boxes[0][1])-10), 0, FONT_SCALE, FONT_COLOR, FONT_THICKNESS)

                # draw.rectangle(boxes[0].tolist(), outline=(255, 0, 0), width=6)
                # draw.text((int(boxes[0][0]), int(boxes[0][3])+3), EMOTION[torch.argmax(emotion)], font=ImageFont.truetype("fonts/gulim.ttc", 20), align ="left")
                    print(f'emotion: {EMOTION[torch.argmax(emotion)]}')
        cv2.imshow('DEMO', frame)

        cv2.waitKey(int(1000/float(args.fps)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

