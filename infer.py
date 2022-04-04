import argparse
import torch
import os
# from model.MobileNet_V3 import *

import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from utils.transforms import NormalizePerImage
import glob
import shutil

parser = argparse.ArgumentParser(description='Emotion inference from cropped face image')
parser.add_argument('--data', type=str, default='/home/keti/storage/dataset/kface_cropped/test')
# parser.add_argument('--data', type=str, default='/home/keti/FER_AR/codes/FER/data/faces_extracted')
parser.add_argument('--workers', default=16, type=int, help='number of data loading workers (default: 16)')
parser.add_argument('--batch-size', default=256, type=int, help='number of mini batch size (default: 256)')
parser.add_argument('--model', type=str, default='/home/keti/FER_AR/codes/FER/saved/effnetv2_s_model_best.pth.tar')
parser.add_argument('--results', type=str, default='/home/keti/FER_AR/codes/FER/results_no_centercrop_aihub')
parser.add_argument('--num-categories', default=7, type=int, help='number of categories (default: 7)')
# parser.add_argument('--image', type=str, default='/home/keti/FER_AR/codes/FER/data/faces_extracted/중립/10.jpg')
parser.add_argument('--image', type=str, default=None)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--if-display-cm', action='store_true', help='If display confusion matrix heatmap')

def main():
    args = parser.parse_args()

    if args.num_categories == 3:
        EMOTION = ['기쁨', '중립', '슬픔']
        EMOTION_eng = ['HAPPY', 'NEUTRAL', 'SAD']
    elif args.num_categories == 7:
        EMOTION = ['기쁨', '당황', '분노', '불안', '상처', '슬픔', '중립']
        EMOTION_eng = ['HAPPY', 'EMBARRASED', 'ANGRY', 'ANXIOUS', 'HURT', 'SAD', 'NEUTRAL']


    top1 = AverageMeter('Acc@1', ':6.2f')
    if 'effnetv2_s' in os.path.basename(args.model):
        import model.efficientNetV2 as models
        model = models.__dict__['effnetv2_s'](num_classes = args.num_categories)
    elif 'effnetv2_m' in os.path.basename(args.model):
        import model.efficientNetV2 as models
        model = models.__dict__['effnetv2_m'](num_classes = args.num_categories)
    elif 'effnetv2_l' in os.path.basename(args.model):
        import model.efficientNetV2 as models
        model = models.__dict__['effnetv2_l'](num_classes = args.num_categories)
    elif 'mobilenet_v3_small' in os.path.basename(args.model):
        import torchvision.models as models
        model = models.__dict__['mobilenet_v3_small'](num_classes=args.num_categories)
    elif 'mobilenet_v3_large' in os.path.basename(args.model):
        import torchvision.models as models
        model = models.__dict__['mobilenet_v3_large'](num_classes=args.num_categories)
    else:
        raise ValueError('Invalid model !!!')

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    checkpoint = torch.load(args.model, map_location=f'cuda:{args.gpu}')
    model.load_state_dict(checkpoint['state_dict'])
    # model = torch.jit.load('checkpoint/mobilenet(emotion7).pth.tar')
    model.eval()
    if args.image == None:
        dataset = datasets.ImageFolder(args.data, transform=
        transforms.Compose([
            transforms.CenterCrop(144),
            transforms.Resize(160),
            transforms.ToTensor(),
            NormalizePerImage()],)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

        # model = mobilenet_v3_large(pretrained=True, num_classes=7)
        # checkpoint = torch.load(args.model)
        # model.load_state_dict(checkpoint['state_dict'])
        y_pred = []
        y_true = []

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for i, (images, target) in enumerate(dataloader):
                    if args.gpu is not None:
                        images = images.cuda(args.gpu, non_blocking=True)
                    if torch.cuda.is_available():
                        target = target.cuda(args.gpu, non_blocking=True)

                    output = model(images)
                    y_pred.extend(torch.argmax(output, dim=1).tolist())
                    y_true.extend(target.tolist())
                    acc1, _ = accuracy(output, target, topk=(1, 2))
                    top1.update(acc1[0], images.size(0))

        cm = confusion_matrix(y_true, y_pred, labels=list(range(args.num_categories)))
        if args.if_display_cm:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = EMOTION_eng)
            disp.plot()
            plt.show()
        np.savetxt(os.path.join(args.results, f"{os.path.basename(args.model.replace('.pth.tar',''))}.csv"), cm, delimiter=",",fmt ='%u')
        with open(os.path.join(args.results, f"{os.path.basename(args.model.replace('.pth.tar',''))}.csv"), 'a') as f:
            f.write(f'\n\ntest acc (%): {round(top1.avg.item(), 2)}\n\n\n')
            # f.write(cm)
        # print(f'acc: {top1.avg}')



    elif args.image == 'fault_finder':
        if os.path.isdir(os.path.join(args.results, f"fault_finder.csv")):
            shutil.rmtree(os.path.join(args.results, f"fault_finder.csv"))

        transform = transforms.Compose([
            transforms.CenterCrop(152),
            transforms.Resize(160),
            transforms.ToTensor(),
            NormalizePerImage()])

        for i, emo in enumerate(EMOTION):
            with open(os.path.join(args.results, f"fault_finder.csv"), 'a') as f:
                f.write(f'\n{emo}\n')
            for path in glob.glob(args.data+f'/{emo}/*.jpg'):
                img = Image.open(path)
                img = torch.unsqueeze(transform(img).cuda(args.gpu, non_blocking=True), 0)

                with torch.no_grad():
                    output = model(img)
                    idx_pred = torch.argmax(output)
                    if i != idx_pred:
                        with open(os.path.join(args.results, f"fault_finder.csv"), 'a') as f:
                            f.write(f'{os.path.basename(path)}\n')

    else:
        transform = transforms.Compose([
            transforms.CenterCrop(152),
            transforms.Resize(160),
            transforms.ToTensor(),
            NormalizePerImage()])
        # img = io.imread(args.image)
        img = Image.open(args.image)
        img = torch.unsqueeze(transform(img).cuda(args.gpu, non_blocking=True), 0)

        with torch.no_grad():
            output = model(img)
            prob = torch.nn.functional.softmax(output)
            emotion_dict = {}
            for i, em in enumerate(EMOTION):
                emotion_dict[em] = round(prob.tolist()[0][i]*100,2)
            # emotion_dict['']
            print(f'결과(확률): {emotion_dict}')
            print(f'감정: {EMOTION[torch.argmax(output)]}')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        # if maxk == 1:                           웃음 두가지 하나로 할때
        #     for i, p in enumerate(pred):
        #         if p == 2:
        #             pred[i]=1
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# def get_confusion_matrix(model ,test_dataset,test_loader):
#     y_pred = []
#     y_true = []
#     model.eval()
#     with torch.no_grad():
#         for image , label in test_loader:
#             output =model(image)
#             _, pred = output.topk(1, 1, True, True)
#             pred = pred.t().cpu()
#             pred = pred.tolist()
#             pred = sum(pred, [])
#             # if (pred == 1 or pred == 2):
#             #     pred = 1
#             y_pred.extend(pred)
#     y_true = test_dataset.targets
#     y_pred_label = np.unique(y_pred)
#     cm = confusion_matrix(y_true , y_pred)
#     conf_matrix = pd.DataFrame(cm , index =y_pred_label  , columns=y_pred_label)
#     #conf_matrix = conf_matrix.loc[[0,1,4],:]        #감정 8개 학습, 감정3개예측할때
#     conf_matrix =conf_matrix.rename(columns = {0 : 'Sad' , 1:'Angry' , 2:'Embarassed' , 3:'Happy', 4:'Neautral', 5:'Hurt', 6:'Anxoius' },
#                                      index = {0 : 'Sad' , 1:'Angry' , 2:'Embarassed' , 3:'Happy', 4:'Neautral', 5:'Hurt', 6:'Anxoius' })
#
#     return conf_matrix



if __name__ == '__main__':
    main()
