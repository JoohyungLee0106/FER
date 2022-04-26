import argparse
import torch
import os
# from model.MobileNet_V3 import *

import model as models
import torchvision.transforms as transforms
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from utils.transforms import transforms_test, NormalizePerImage
import glob
import shutil

parser = argparse.ArgumentParser(description='Emotion inference from cropped face image')
parser.add_argument('--data', type=str, default='data/aihub/test')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 16)')
parser.add_argument('--batch-size', default=4, type=int, help='number of mini batch size (default: 256)')
parser.add_argument('--model', type=str, default='ex_model_best.pth.tar')
parser.add_argument('--results', type=str, default='results')
parser.add_argument('--image', type=str, default=None)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--if-display-cm', action='store_true', help='If display confusion matrix heatmap')

def main():
    args = parser.parse_args()
    if os.path.isdir(args.results):
        shutil.rmtree(args.results)
    os.mkdir(args.results)

    top1 = AverageMeter('Acc@1', ':6.2f')

    # args.model 에서 모델weight 을 불러와서 args.gpu에 카피.
    checkpoint = torch.load(args.model, map_location=f'cuda:{args.gpu}')
    args.emotions = list(checkpoint['classes'])
    model = models.__dict__[checkpoint['model']](num_classes=len(args.emotions))

    torch.cuda.set_device(args.gpu)
    
    # 명시된 gpu로의 모델 카피
    model = model.cuda(args.gpu)
    # 위에서 카피된 모델weight를 gpu로 카피된 모델에 로딩시킴
    model.load_state_dict(checkpoint['state_dict'])
    # 훈련이 아니므로 eval 모드로 전환
    model.eval()
    if args.image == None:
        dataset = datasets.ImageFolder(args.data, transform=transforms_test())
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

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

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(args.emotions))))
        if args.if_display_cm:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels = args.emotions)
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

        transform = transforms_test()

        for i, emo in enumerate(args.emotions):
            with open(os.path.join(args.results, f"fault_finder.csv"), 'a') as f:
                f.write(f'\n{emo}\n')
            for path in glob.glob(os.path.join(args.data, f'{emo}/*.jpg')):
                img = Image.open(path)
                img = torch.unsqueeze(transform(img).cuda(args.gpu, non_blocking=True), 0)

                with torch.no_grad():
                    output = model(img)
                    idx_pred = torch.argmax(output)
                    if i != idx_pred:
                        with open(os.path.join(args.results, f"fault_finder.csv"), 'a') as f:
                            f.write(f'{os.path.basename(path)}\n')

    else:
        transform = transforms_test()
        # img = io.imread(args.image)
        img = Image.open(args.image)
        img = torch.unsqueeze(transform(img).cuda(args.gpu, non_blocking=True), 0)

        with torch.no_grad():
            output = model(img)
            prob = torch.nn.functional.softmax(output)
            emotion_dict = {}
            for i, em in enumerate(args.emotions):
                emotion_dict[em] = round(prob.tolist()[0][i]*100,2)
            # emotion_dict['']
            print(f'결과(확률): {emotion_dict}')
            print(f'감정: {args.emotions[torch.argmax(output)]}')


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


if __name__ == '__main__':
    main()
