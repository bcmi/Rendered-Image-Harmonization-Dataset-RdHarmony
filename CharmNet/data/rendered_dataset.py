import os.path
import torch
import torchvision.transforms.functional as tf
from data.base_dataset import BaseDataset, get_transform
#from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as transforms
import random

class RenderedDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.image_paths = []
        self.unharm_GT_labels = []
        self.harmed_GT_labels = []
        self.isTrain = opt.isTrain
        if opt.isTrain==True:
            print('loading training file in rendered domain: ')
            self.trainfile = os.path.join(opt.dataset_root, opt.render_novel_list)
            self.keep_background_prob = 0.05
            with open(self.trainfile,'r') as f:
                for line in f.readlines():
                    line = line.rstrip().split('\t')
                    unharm_label = line[1].split('_')
                    for idx in range(len(unharm_label)):
                        unharm_label[idx] = float(unharm_label[idx])
                    self.unharm_GT_labels.append(unharm_label)
                    self.harmed_GT_labels.append(int(line[2]) - 1)
                    self.image_paths.append(os.path.join(opt.dataset_root,line[0].rstrip()))
        self.transform = get_transform(opt)
        self.input_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
            ])

    def __getitem__(self, index):
        sample = self.get_sample(index)
        self.check_sample_types(sample)
        sample = self.augment_sample(sample)
        comp = self.input_transform(sample['image'])
        real = self.input_transform(sample['real'])
        mask = sample['mask'].astype(np.float32)
        output = {
            'comp': comp,
            'mask': mask[np.newaxis, ...].astype(np.float32),
            'real': real,
            'img_path':sample['img_path'],
            'unharm_GT_label': sample['unharm_GT_label'],
            'harmed_GT_label': sample['harmed_GT_label']
        }
        return output

    def check_sample_types(self, sample):
        assert sample['comp'].dtype == 'uint8'
        if 'real' in sample:
            assert sample['real'].dtype == 'uint8'

    def augment_sample(self, sample):
        if self.transform is None:
            return sample
        additional_targets = {target_name: sample[target_name]
                              for target_name in self.transform.additional_targets.keys()}

        valid_augmentation = False
        while not valid_augmentation:
            aug_output = self.transform(image=sample['comp'], **additional_targets)
            valid_augmentation = self.check_augmented_sample(sample, aug_output)

        for target_name, transformed_target in aug_output.items():
            sample[target_name] = transformed_target

        return sample

    def check_augmented_sample(self, sample, aug_output):
        if self.keep_background_prob < 0.0 or random.random() < self.keep_background_prob:
            return True

        return aug_output['mask'].sum() > 1.0

    def get_sample(self, index):
        path = self.image_paths[index]
        name_parts_under=path.split('_')
        target_path = self.image_paths[index].replace('composite_images','real_images')
        target_path = target_path.replace(('_'+name_parts_under[-1]),'.jpg')
        name_parts=path.split('-')
        mask_path = self.image_paths[index].replace('composite_images','masks')
        mask_path = mask_path.replace(('-'+name_parts[-1]),'.png')

        unharm_GT_label = np.array(self.unharm_GT_labels[index])
        harmed_GT_label = self.harmed_GT_labels[index]

        comp = cv2.imread(path)
        comp = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
        real = cv2.imread(target_path)
        real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path)
        mask = mask[:, :, 0].astype(np.float32) / 255.
        mask = mask.astype(np.uint8)


        return {'comp': comp, 'mask': mask, 'real': real,'img_path':path, 'unharm_GT_label': unharm_GT_label, 'harmed_GT_label': harmed_GT_label}

    def __len__(self):
        """Return the total number of images."""
        return len(self.image_paths)

