import numpy as np
import torch
from torch import tensor
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# 数据格式：anno: (tensor([0, 1]), tensor([[x1, y1, x2, y2], [364., 167., 487., 402.], [350.,  85., 488., 401.]]), {scale, pad_loc})
# 调用格式：img, anno = self.transforms(img, anno)


class Resize(object):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def __call__(self, img, anno):
        gts = anno[1].numpy()
        np_img = np.asarray(img)
        scale = self.h / np_img.shape[0]

        # get BoundingBoxesOnImage
        bbs = []
        for gt in gts:
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)

        self.seq = iaa.Sequential([
            iaa.Resize({"height": self.h, "width": self.w})
        ])

        # apply augment
        image_aug = self.seq.augment_image(np_img)
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes

        gts = []
        for bb in bbs_aug:
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])

        gts = torch.from_numpy(np.array(gts))
        anno[2]["scale"] = scale
        return image_aug, (anno[0], gts, anno[2])


class Rotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, anno):
        gts = anno[1].numpy()
        np_img = np.asarray(img)

        # get BoundingBoxesOnImage
        bbs = []
        for gt in gts:
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)
        self.seq = iaa.Sequential([
            iaa.Affine(rotate=self.degree)
        ])
        # apply augment
        image_aug = self.seq.augment_image(np_img)
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes

        gts = []
        for bb in bbs_aug:
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])

        gts = torch.from_numpy(np.array(gts))
        return image_aug, (anno[0], gts, anno[2])

class RandomCropAndPad(object):	
    def __init__(self, percent=0.1):	
        self.percent = percent	

    def __call__(self, img, anno):	
        labels = anno[0].numpy()	
        gts = anno[1].numpy()	
        np_img = np.asarray(img)	

        # get BoundingBoxesOnImage	
        bbs = []	
        for gt in gts:	
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))	

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)	
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)	

        r = np.random.sample()	
        p = -self.percent + 2 * self.percent * r	

        self.seq = iaa.Sequential([	
            iaa.CropAndPad(	
                percent=p,	
                keep_size=False	
            )	
        ])	

         # apply augment	
        image_aug = self.seq.augment_image(np_img)	
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes	

        gts = []	
        for bb in bbs_aug:	
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])	

        gts = torch.from_numpy(np.array(gts))	
        return image_aug, (anno[0], gts, anno[2])

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, anno):
        gts = anno[1].numpy()
        np_img = np.asarray(img)

        # get BoundingBoxesOnImage
        bbs = []
        for gt in gts:
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)

        degree = np.random.randint(low=-self.degree, high=self.degree)
        self.seq = iaa.Sequential([
            iaa.Affine(rotate=degree)
        ])
        # apply augment
        image_aug = self.seq.augment_image(np_img)
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes

        gts = []
        for bb in bbs_aug:
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])

        gts = torch.from_numpy(np.array(gts))
        return image_aug, (anno[0], gts, anno[2])


class RandomTranslatePx(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, img, anno):
        gts = anno[1].numpy()
        np_img = np.asarray(img)

        # get BoundingBoxesOnImage
        bbs = []
        for gt in gts:
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)

        x = np.random.randint(-self.x, self.x)
        y = np.random.randint(-self.y, self.y)
        self.seq = iaa.Sequential([
            iaa.Affine(translate_px={"x": x, "y": y})
        ])

        # apply augment
        image_aug = self.seq.augment_image(np_img)
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes

        gts = []
        for bb in bbs_aug:
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])

        gts = torch.from_numpy(np.array(gts))
        return image_aug, (anno[0], gts, anno[2])


class RandomTranslatePc(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __call__(self, img, anno):
        gts = anno[1].numpy()
        np_img = np.asarray(img)

        # get BoundingBoxesOnImage
        bbs = []
        for gt in gts:
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)

        x = np.random.randint(-self.x, self.x)
        y = np.random.randint(-self.y, self.y)
        self.seq = iaa.Sequential([
            iaa.Affine(translate_percent={"x": x, "y": y})
        ])
        # apply augment
        image_aug = self.seq.augment_image(np_img)
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes

        gts = []
        for bb in bbs_aug:
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])

        gts = torch.from_numpy(np.array(gts))
        return image_aug, (anno[0], gts, anno[2])


class RandomFlipLeftRight(object):
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, img, anno):
        if np.random.random_sample() > self.probability:
            return img, anno

        gts = anno[1].numpy()
        np_img = np.asarray(img)

        # get BoundingBoxesOnImage
        bbs = []
        for gt in gts:
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)

        self.seq = iaa.Sequential([
            iaa.Fliplr(1)  # 水平翻转图像
        ])

        # apply augment
        image_aug = self.seq.augment_image(np_img)
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes

        gts = []
        for bb in bbs_aug:
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])

        gts = torch.from_numpy(np.array(gts))
        return image_aug, (anno[0], gts, anno[2])


class RandomFlipUpDown(object):
    def __init__(self, probability):
        self.probability = probability

    def __call__(self, img, anno):
        if np.random.random_sample() > self.probability:
            return img, anno

        gts = anno[1].numpy()
        np_img = np.asarray(img)

        # get BoundingBoxesOnImage
        bbs = []
        for gt in gts:
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)

        self.seq = iaa.Sequential([
            iaa.Flipud(1)
        ])

        # apply augment
        image_aug = self.seq.augment_image(np_img)
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes

        gts = []
        for bb in bbs_aug:
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])

        gts = torch.from_numpy(np.array(gts))
        return image_aug, (anno[0], gts, anno[2])


class RandomSaltPepperNoise(object):
    def __init__(self, SNR, probability):
        self.SNR = SNR
        self.probability = probability

    def __call__(self, img, anno):
        if np.random.random_sample() > self.probability:
            return img, anno

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

        return image_aug, anno

# super light-weight wrapper

class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.transform = transforms.Normalize(mean=mean, std=std)

    def __call__(self, img, anno):
        img = self.transform(img)
        return img, anno

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, img, anno):
        img = self.transform(img)
        return img, anno

class Compose(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img, anno):
        for tran in self.transform_list:
            img, anno = tran(img, anno)

        return img, anno

class RandomChoice(object):
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, img, anno):
        p = np.random.sample()
        choice = int(len(self.transform_list) * p)

        tran = self.transform_list[choice]
        img, anno = tran(img, anno)

        return img, anno

class RandomContrast(object):
    def __init__(self, contrast=0.1):
        # contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast] or the given [min, max]
        self.transform = transforms.ColorJitter(contrast=contrast)

    def __call__(self, img, anno):
        img = self.transform(img)
        return img, anno

class Contrast(object):
    def __init__(self, contrast=1):
        self.contrast = contrast

    def __call__(self, img, anno):
        img = transforms.functional.adjust_contrast(img, self.contrast)
        return img, anno

class AutoContrast(object):
    def __init__(self, cutoff=0, ignore=None):
        self.cutoff = cutoff
        self.ignore = ignore

    def __call__(self, img, anno):
        img = ImageOps.autocontrast(img, self.cutoff, self.ignore)
        return img, anno


class AutoLevel(object):
    def __init__(self, min_level_rate=1., max_level_rate=1.):
        self.min_level_rate = min_level_rate
        self.max_level_rate = max_level_rate

    def __call__(self, img, anno):
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
        return img, anno

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


class Pad(object):
    def __init__(self, position="center"):
        # {'uniform', 'normal', 'center', 'left-top', 'left-center', 'left-bottom', 'center-top', 'center-center',\
        #             'center-bottom', 'right-top', 'right-center', 'right-bottom'}
        self.position = position

    def __call__(self, img, anno):
        gts = anno[1].numpy()
        np_img = np.asarray(img)

        # get BoundingBoxesOnImage
        bbs = []
        for gt in gts:
            bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

        bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
        # draw_img = bbs_on_img.draw_on_image(np_img, size=2)

        height = np_img.shape[0]
        width = np_img.shape[1]
        if height >= width:
            length = height
        else:
            length = width
        
        # get pad loc, up, down, left, right
        pad_loc = [0, 0, 0, 0]
        if self.position == "center":
            if height >= width:
                pad_loc[2] = pad_loc[3] = (height - width) / 2
            else:
                pad_loc[0] = pad_loc[1] = (width - height) / 2
            
        # position is must, because the bbs and images are augmented separately
        self.seq = iaa.Sequential([
            iaa.PadToFixedSize(width=length, height=length, position=self.position)
        ])

        # apply augment
        image_aug = self.seq.augment_image(np_img)
        bbs_aug = self.seq.augment_bounding_boxes(bbs_on_img).bounding_boxes

        gts = []
        for bb in bbs_aug:
            gts.append([bb.x1, bb.y1, bb.x2, bb.y2])

        gts = torch.from_numpy(np.array(gts))
        anno[2]["pad_loc"] = pad_loc
        return image_aug, (anno[0], gts, anno[2])


def show_bbs(img, anno):
    gts = anno[1].numpy()
    np_img = np.asarray(img)

    bbs = []
    for gt in gts:
        bbs.append(BoundingBox(x1=gt[0], y1=gt[1], x2=gt[2], y2=gt[3]))

    bbs_on_img = BoundingBoxesOnImage(bbs, shape=np_img.shape)
    draw_img = bbs_on_img.draw_on_image(np_img, size=2)
    ia.imshow(draw_img)

'''
anno = (tensor([0, 1]), tensor([[64., 30., 187., 150.], [20.,  85., 188., 101.]]))
image = Image.open('./lena.jpg')

show_bbs(image, anno)

image_aug, anno_aug = FlipUpDown(0.5)(image, anno)
show_bbs(image_aug, anno_aug)
image_aug, anno_aug = FlipLeftRight(0.5)(image, anno)
show_bbs(image_aug, anno_aug)
'''
# image_aug, anno_aug = Resize(512, 700)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = Rotate(-45)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = AutoRotate(45)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = TranslatePx(20, 40)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = TranslatePc(0.2, 0.3)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = SaltPepperNoise(0.7, 0.9)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = Normalize()(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = RandomContrast()(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = AutoContrast(10, 5)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = AutoLevel(2, 2)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = Resize(256, 512)(image, anno)
# show_bbs(image_aug, anno_aug)
# image_aug, anno_aug = Pad()(image_aug, anno_aug)
# show_bbs(image_aug, anno_aug)

# todo random init cut

# print(anno_aug[1])
