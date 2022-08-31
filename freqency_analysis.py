data_dir = '../../Uformer-main/data/'
import cv2
import torch
import matplotlib.pyplot as plt
import torch.fft as fft
import numpy as np
haze =  cv2.cvtColor(cv2.imread(data_dir+'val/fog/1_2.png'), cv2.COLOR_BGR2RGB)
gt = cv2.cvtColor(cv2.imread(data_dir+'val/gt/1_2.png'), cv2.COLOR_BGR2RGB)
pred = cv2.imread('results/endoscopic/dehazeformer-s/imgs/1_2.png')
pred = pred[:,512:,:]

pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
plt.imshow(haze)
plt.show()
plt.imshow(gt)
plt.show()
plt.imshow(pred)
plt.show()
h,w,c = haze.shape
def show_fft(haze=haze,title = 'haze_fft'):
    haze_fft = fft.fftn(torch.tensor(haze),dim=(0,1))
    haze_fft = torch.roll(haze_fft,(h//2,w//2),dims=(0,1))
    haze_fft_show = torch.log(1+torch.abs(haze_fft))
    haze_fft_show = (haze_fft_show - haze_fft_show.min())/(haze_fft_show.max() - haze_fft_show.min())*255
    plt.imshow(haze_fft_show.numpy().astype(np.uint8))
    plt.title(title)
    plt.show()

show_fft(gt,'gt_fft')
show_fft(pred,'pred_fft')