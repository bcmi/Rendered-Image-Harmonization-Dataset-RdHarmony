from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time

import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html,util

from skimage import data, img_as_float
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr

import torchvision.transforms.functional as tf

sub_datasets = ['HCOCO','HAdobe5k','HFlickr','Hday2night']
sub_datasets_ind = ['c','a','f','d']
test_dirs = ['0to5','5to15','15to90']
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 1
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset_mode = opt.dataset_mode
    dataset = create_dataset(opt,dataset_mode)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    print('the number of test image: ', len(dataset))
    
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    file_name_output = web_dir+'/'+TIMESTAMP+'_'+opt.epoch+'.txt'
    file_op = open(file_name_output,'a')
    file_name_result = web_dir+'/'+TIMESTAMP+'_'+opt.epoch+'_res.txt'
    file_res = open(file_name_result,'a')
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = str(data['img_path'])
        raw_name = img_path.replace(('[\''),'')
        raw_name = raw_name.replace(('.jpg\']'),'.jpg')
        mask_name = raw_name.replace('/composite_images/','/masks/')
        parts = mask_name.split('_')
        mask_name = parts[0]+'_'+parts[1]+'.png'
        raw_name = raw_name.split('/')[-1]
        image_name = '%s' % raw_name
        save_path = os.path.join(web_dir,'images/',image_name)
        for label, im_data in visuals.items():
            if label=='tgt_harm':
                output = util.tensor2im(im_data)
                output =np.array(output, dtype=np.float32)
            if label=='tgt_real':
                real=util.tensor2im(im_data)
                real=np.array(real, dtype=np.float32)
            if label=='tgt_comp':
                comp=util.tensor2im(im_data)
                comp=np.array(comp, dtype=np.float32)
            if label=='tgt_mask':
                mask_tensor = im_data

        mask = Image.open(mask_name)
        mask = mask.convert('1')
        mask = mask.resize([256,256], Image.BICUBIC)
        mask = np.array(mask, dtype=np.uint8)
        fore_area = np.sum(np.sum(mask,axis=0),axis=0)
        mask = mask[...,np.newaxis]

        pred_tensor = tf.to_tensor(output).unsqueeze(0).cuda()
        real_tensor = tf.to_tensor(real).unsqueeze(0).cuda()
        mask_tensor = tf.to_tensor(mask).unsqueeze(0).cuda()

        fmse_score_op=mse(output*mask,real*mask)*256*256/fore_area
        mse_score_op = mse(output,real)
        psnr_score_op = psnr(real,output,data_range=output.max() - output.min())

        print('%s | mse %0.4f | fmse %0.4f | psnr %0.4f' % (image_name,mse_score_op,fmse_score_op,psnr_score_op))
        file_op.writelines('%s\t%f\t%f\t%f\n' % (image_name,mse_score_op,fmse_score_op,psnr_score_op))

    webpage.save()  # save the HTML
    file_op.close()

    f_name=file_name_output
    with open(f_name,'r') as f:
        name_mse=[line.rstrip() for line in f.readlines()]
    mse_all=[]
    psnr_all=[]
    fmse_all=[]
    for pp in range(0,len(name_mse)):
        mse_all.append(name_mse[pp].split('\t')[1])
        psnr_all.append(name_mse[pp].split('\t')[2])
        fmse_all.append(name_mse[pp].split('\t')[3])
    mse_all=np.array(mse_all).astype(np.float)
    psnr_all=np.array(psnr_all).astype(np.float)
    fmse_all=np.array(fmse_all).astype(np.float)

    print('WHOLE: %0.4f/%0.4f/%0.4f' % (np.mean(mse_all),np.mean(fmse_all),np.mean(psnr_all)))
    file_res.writelines('WHOLE: %s/%s/%s\n' %(str(np.mean(mse_all)),str(np.mean(fmse_all)),str(np.mean(psnr_all))))
    for jj in range(0,len(sub_datasets)):
        dataset_ind = sub_datasets_ind[jj]
        mse_avg=[]
        psnr_avg=[]
        fmse_avg=[]
        for mm in range(0,len(name_mse)):
            name = name_mse[mm].split('\t')[0]
            if name[0]==dataset_ind:
                mse_avg.append(name_mse[mm].split('\t')[1])
                psnr_avg.append(name_mse[mm].split('\t')[2])
                fmse_avg.append(name_mse[mm].split('\t')[3])
        mse_avg=np.array(mse_avg).astype(np.float)
        psnr_avg=np.array(psnr_avg).astype(np.float)
        fmse_avg=np.array(fmse_avg).astype(np.float)
        print('%s: %0.4f/%0.4f/%0.4f' % (sub_datasets[jj],np.mean(mse_avg),np.mean(fmse_avg),np.mean(psnr_avg)))
        file_res.writelines('%s: %s/%s/%s\n' %(str(sub_datasets[jj]),str(np.mean(mse_avg)),str(np.mean(fmse_avg)),str(np.mean(psnr_avg))))

