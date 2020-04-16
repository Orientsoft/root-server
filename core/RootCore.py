import os
import io

from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageMath
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

from retinanet import model as detection_model
from retinanet import transforms as detection_aug
from resnet import transforms as score_aug
from resnet import resnet as score_model

# consts
CLASSES = {
    'root': 0,
    'tooth': 1,
    'treated-root': 2,
    'treated-tooth': 3,
    'uncertain': 4
}

BBOX_COLOR_NAME = 'blue'
SCORE_COLOR_NAME = 'lime'
BBOX_LINE_WIDTH = 2

IMAGE_SIZE = 512
MAX_DETECTIONS = 12
NUM_CLASSES = 5

STD = 7.549563
MEAN = 78.416294

SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
SCALE = 1.2

class RootCore():
    def __init__(self, detection_model_path, score_model_path, private_key, device_name='cpu'):
        self.detection_model_path = detection_model_path
        self.score_model_path = score_model_path
        self.private_key = private_key
        self.device_name = device_name

        root_core_path, _ = os.path.split(os.path.abspath(__file__))
        self.font = ImageFont.truetype(os.path.join(root_core_path, 'resource/arial.ttf'), 24)

        self.device = torch.device(self.device_name)

        self.detect_trans = detection_aug.Compose([
            detection_aug.Pad(),
            detection_aug.Resize(IMAGE_SIZE, IMAGE_SIZE),
            detection_aug.AutoLevel(min_level_rate=1, max_level_rate=1),
            detection_aug.AutoContrast(),
            detection_aug.Contrast(1.25),
            detection_aug.ToTensor()
        ])
        self.score_trans = transforms.Compose([
            score_aug.AutoLevel(),
            score_aug.AutoContrast(),
            score_aug.Contrast(contrast=1.2),
            score_aug.Pad(), # pad to square
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor()
        ])

        # TODO : decrypt model with private_key

        self.detection_net = detection_model.resnet50(num_classes=NUM_CLASSES)
        self.score_net = score_model.resnet50(num_classes=1)

        self.detection_net.load_state_dict(torch.load(self.detection_model_path))
        self.score_net.load_state_dict(torch.load(self.score_model_path))
        

        self.detection_net.to(self.device)
        self.score_net.to(self.device)

    def detect(self, picture_path):
        self.detection_net.eval()
        self.detection_net.set_nms(NMS_THRESHOLD)

        img = Image.open(picture_path)

        if img.mode == 'I':
            img = self._convert_I16_to_L(img)

        img = img.convert('RGB')

        data = self.detect_trans(img, (torch.tensor([]), torch.tensor([]), {}))

        scale = data[1][2]["scale"]
        pad_loc = data[1][2]["pad_loc"]

        with torch.no_grad():
            scores, labels, boxes = self.detection_net(data[0].permute(0, 1, 2).to(self.device).float().unsqueeze(dim=0))
            
            scores = scores.cpu().numpy()
            labels = labels.cpu().numpy()
            boxes = boxes.cpu().numpy()

            # correct boxes
            boxes /= scale
            for box in boxes:
                box[1] -= pad_loc[0]
                box[3] -= pad_loc[0]
                box[0] -= pad_loc[2]
                box[2] -= pad_loc[2]

            # SCORE_THRESHOLD
            indices = np.where(scores > SCORE_THRESHOLD)[0]

            if indices.shape[0] > 0:
                scores = scores[indices]
                scores_sort = np.argsort(-scores)[:MAX_DETECTIONS]

                image_boxes = boxes[indices[scores_sort], :]
                image_scores = scores[scores_sort]
                image_labels = labels[indices[scores_sort]]

                '''
                image_detections = np.concatenate([
                    image_boxes,
                    np.expand_dims(image_scores, axis=1),
                    np.expand_dims(image_labels, axis=1)
                ], axis=1)

                return image_detections
                '''

                image_detections = []

                for box, label in zip(image_boxes, image_labels):
                    if label == CLASSES['treated-root']:
                        image_detections.append(box.tolist())
                print(image_detections)
                return image_detections

            return []

    def bbox_score(self, picture_path, bboxes):
        if len(bboxes) == 0:
            return []
        
        img = Image.open(picture_path)
        
        if img.mode == 'I':
            img = self._convert_I16_to_L(img)
        
        img = img.convert('RGB')
        
        crops = torch.stack([self.score_trans(img.crop(bbox)) for bbox in bboxes])

        self.score_net.training = False
        self.score_net.eval()

        with torch.no_grad():
            results = self.score_net(crops.to(self.device))

            score_bboxes = np.concatenate([
                np.array(bboxes),
                results.cpu().numpy() * STD + MEAN
            ], axis=1)

            return score_bboxes.tolist()

        '''
        for bbox in bboxes:
            crop = img.crop(bbox)
            crop = self.score_trans(crop)

            results = self.score_net(crop.to(self.device).unsqueeze(dim=0))
        '''

    def score(self, picture_path):
        bboxes = self.detect(picture_path)

        if len(bboxes) > 0:
            score_bboxes = self.bbox_score(picture_path, bboxes)
            return score_bboxes
        else:
            return []

    def draw_score_bboxes(self, picture_path, score_bboxes, bbox_color_name=BBOX_COLOR_NAME, score_color_name=SCORE_COLOR_NAME, bbox_line_width=BBOX_LINE_WIDTH, draw_score=True):
        img = Image.open(picture_path)
        
        if img.mode == 'I':
            img = self._convert_I16_to_L(img)
        
        img = img.convert('RGB')
            
        draw = ImageDraw.Draw(img)

        for score_bbox in score_bboxes:
            bbox = score_bbox[:4]
            score = score_bbox[4]

            draw.rectangle(
                bbox,
                outline=ImageColor.getrgb(bbox_color_name),
                width=bbox_line_width
            )

            if draw_score:
                draw.text(
                    bbox[:2],
                    '{:.2f}'.format(score),
                    font=self.font,
                    fill=ImageColor.getrgb(score_color_name)
                )

        del draw

        imgArr = io.BytesIO()
        img.save(imgArr, format='JPEG')

        return imgArr.getvalue()
    
    def _convert_I16_to_L(self, i16_img):
        im2 = ImageMath.eval('im/256', {'im':i16_img}).convert('L')
        
        return im2
