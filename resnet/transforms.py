import numpy as np
import torch
from torch import tensor
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

class Pad(object):
    def __init__(self, position="center"):
        # {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center',\
        #             'center-bottom', 'right-top', 'right-center', 'right-bottom'}
        self.position = position

    def __call__(self, img):
        np_img = np.asarray(img)

        height = np_img.shape[0]
        width = np_img.shape[1]
        if height >= width:
            length = height
        else:
            length = width
            
        # position is must, because the bbs and images are augmented separately
        self.seq = iaa.Sequential([
            iaa.PadToFixedSize(width=length, height=length, position=self.position)
        ])

        # apply augment
        image_aug = self.seq.augment_image(np_img)

        return image_aug

class RandomSaltPepperNoise(object):
    def __init__(self, SNR, probability):
        self.SNR = SNR
        self.probability = probability

    def __call__(self, img):
        if np.random.random_sample() > self.probability:
            return img

        np_img = np.asarray(img)
        image_aug = np_img.copy()
        noise_num = int((1 - self.SNR) * image_aug.shape[0] * image_aug.shape[1])
        h, w, c = image_aug.shape
        for i in range(noise_num):
            x = np.random.randint(0, h)
            y = np.random.randint(0, w)
            if np.random.randint(0, 1) == 0:
                image_aug[x, y, :] = 255
            else:
                image_aug[x, y, :] = 0

        return image_aug
    
class AutoContrast(object):
    def __init__(self, cutoff=0, ignore=None):
        self.cutoff = cutoff
        self.ignore = ignore

    def __call__(self, img):
        img = ImageOps.autocontrast(img, self.cutoff, self.ignore)
        return img

class RandomContrast(object):
    def __init__(self, contrast=0.1):
        # contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]
        self.transform = transforms.ColorJitter(contrast=contrast)

    def __call__(self, img):
        img = self.transform(img)
        return img

class Contrast(object):
    def __init__(self, contrast=1):
        self.contrast = contrast

    def __call__(self, img):
        img = transforms.functional.adjust_contrast(img, self.contrast)
        return img

class AutoLevel(object):
    def __init__(self, min_level_rate=1., max_level_rate=1.):
        self.min_level_rate = min_level_rate
        self.max_level_rate = max_level_rate

    def __call__(self, img):
        img = np.asarray(img)
        h, w, d = img.shape
        newimg = np.zeros([h, w, d])
        for i in range(d):
            img_hist = self.compute_hist(img[:, :, i])
            min_level = self.compute_min_level(img_hist, self.min_level_rate, h * w)
            max_level = self.compute_max_level(img_hist, self.max_level_rate, h * w)
            newmap = self.linear_map(min_level, max_level)
            if newmap.size == 0:
                continue
            for j in range(h):
                newimg[j, :, i] = newmap[img[j, :, i]]
        img = Image.fromarray(np.uint8(newimg))
        return img

    def compute_hist(self, img):
        h, w = img.shape
        hist, bin_edge = np.histogram(img.reshape(1, w * h), bins=list(range(257)))
        return hist

    def compute_min_level(self, hist, rate, pnum):
        sum = 0
        for i in range(256):
            sum += hist[i]
            if sum >= (pnum * rate * 0.01):
                return i

    def compute_max_level(self, hist, rate, pnum):
        sum = 0
        for i in range(256):
            sum += hist[255 - i]
            if sum >= (pnum * rate * 0.01):
                return 255 - i

    def linear_map(self, min_level, max_level):
        if min_level >= max_level:
            return []
        else:
            newmap = np.zeros(256)
            for i in range(256):
                if i < min_level:
                    newmap[i] = 0
                elif i > max_level:
                    newmap[i] = 255
                else:
                    newmap[i] = (i - min_level) / (max_level - min_level) * 255
            return newmap