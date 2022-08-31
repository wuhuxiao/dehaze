import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torch.utils.data import DataLoader
from collections import OrderedDict

from utils import AverageMeter, write_img, chw_to_hwc
from datasets.loader import PairLoader
from models import *
import sys
import numpy as np
import cv2

isDebug = True if sys.gettrace() else False

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='dehazeformer-s', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='../../Uformer-main/data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--dataset', default='endoscopic', type=str, help='dataset name')
parser.add_argument('--exp', default='endoscopic', type=str, help='experiment setting')
args = parser.parse_args()

if isDebug:
    args.num_workers = 0


def single(save_dir):
    state_dict = torch.load(save_dir)['state_dict']
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v

    return new_state_dict


def test(test_loader, network, result_dir):
    PSNR = AverageMeter()
    SSIM = AverageMeter()

    torch.cuda.empty_cache()

    network.eval()

    os.makedirs(os.path.join(result_dir, 'imgs'), exist_ok=True)
    f_result = open(os.path.join(result_dir, 'results.csv'), 'w')

    for idx, batch in enumerate(test_loader):
        input = batch['source'].cuda()
        target = batch['target'].cuda()

        filename = batch['filename'][0]

        with torch.no_grad():
            output = network(input).clamp_(-1, 1)

            # [-1, 1] to [0, 1]
            output = output * 0.5 + 0.5
            target = target * 0.5 + 0.5
            input = input * 0.5 + 0.5

            psnr_val = 10 * torch.log10(1 / F.mse_loss(output, target)).item()
            psnr_base = 10 * torch.log10(1 / F.mse_loss(input, target)).item()

            _, _, H, W = output.size()
            down_ratio = max(1, round(min(H, W) / 256))  # Zhou Wang
            ssim_val = ssim(F.adaptive_avg_pool2d(output, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()

            ssim_base = ssim(F.adaptive_avg_pool2d(input, (int(H / down_ratio), int(W / down_ratio))),
                            F.adaptive_avg_pool2d(target, (int(H / down_ratio), int(W / down_ratio))),
                            data_range=1, size_average=False).item()


        PSNR.update(psnr_val)
        SSIM.update(ssim_val)

        print('Test: [{0}]\t'
              'PSNR: {psnr.val:.02f} ({psnr.avg:.02f})\t'
              'SSIM: {ssim.val:.03f} ({ssim.avg:.03f})'
              .format(idx, psnr=PSNR, ssim=SSIM))

        f_result.write('%s,%.02f,%.03f\n' % (filename, psnr_val, ssim_val))

        out_img = chw_to_hwc(output.detach().cpu().squeeze(0).numpy())
        filename = os.path.join(result_dir, 'imgs', filename)
        input = chw_to_hwc(input.detach().cpu().squeeze(0).numpy())
        target = chw_to_hwc(target.detach().cpu().squeeze(0).numpy())
        img = np.concatenate((input, target, out_img), axis=1)

        img = np.round((img[:, :, ::-1].copy() * 255.0)).astype('uint8')
        text =  f'psnr: {psnr_val:.4f}  ssim:{ssim_val:.4f}'
        text1 =  f'identity psnr: {psnr_base:.4f}  ssim:{ssim_base:.4f}'

        cv2.putText(img, text, (5, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 200), 1)
        cv2.putText(img, text1, (5, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 200), 1)
        cv2.imwrite(filename, img)

    f_result.close()

    os.rename(os.path.join(result_dir, 'results.csv'),
              os.path.join(result_dir, '%.02f | %.04f.csv' % (PSNR.avg, SSIM.avg)))


if __name__ == '__main__':
    network = eval(args.model.replace('-', '_'))()
    network.cuda()
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = args.data_dir
    test_dataset = PairLoader(dataset_dir, 'val', 'test')
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True,
                             drop_last=False)

    result_dir = os.path.join(args.result_dir, args.dataset, args.model)
    test(test_loader, network, result_dir)
