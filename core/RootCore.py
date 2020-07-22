import os
import io

from PIL import Image, ImageDraw, ImageColor, ImageFont, ImageMath
import numpy as np

import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
from torchvision.ops import nms

from retinanet import model as detection_model
from retinanet import transforms as detection_aug
from retinanet.utils import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from retinanet.anchors import Anchors
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
VINO_CROP_SIZE = 224
MAX_DETECTIONS = 12
NUM_CLASSES = 5

STD = 7.549563
MEAN = 78.416294

SCORE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.5
SCALE = 1.2

DETECTION_VINO_DEVICE = 'CPU'
SCORE_VINO_DEVICE = 'CPU'

class DetectionPostProcessor():
    def __init__(self, nms, score):
        self.anchors = Anchors()
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.nms = nms
        self.score_threshold = score
        
    def process(self, img_batch, regression, classification):
        anchors = self.anchors(img_batch)

        transformed_anchors = self.regressBoxes(anchors, regression)
        transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

        scores = torch.max(classification, dim=2, keepdim=True)[0]
        scores_over_thresh = (scores>0.05)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just return
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, :].squeeze(), self.nms)

        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

        return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]
    
    def export(self, scores, labels, boxes, scale, pad_loc):
        # correct boxes
        '''
        w_scale = w / ori_w
        h_scale = h / ori_h
        boxes[:, 0::2] /= w_scale
        boxes[:, 1::2] /= h_scale
        '''
        
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
            
            if scores_sort.shape[0] == 1:
                image_boxes = image_boxes.unsqueeze(0)
                image_scores = image_scores.unsqueeze(0)
                image_labels = image_labels.unsqueeze(0)

            image_detections = []

            for box, label in zip(image_boxes, image_labels):
                if label == CLASSES['treated-root']:
                    image_detections.append(box.tolist())

            return image_detections

        return []

class RootCore():
    def __init__(self, detection_model_path, score_model_path, private_key, device_name='cpu', backend='pytorch', det_vino_device=DETECTION_VINO_DEVICE, score_vino_device=SCORE_VINO_DEVICE):
        self.detection_model_path = detection_model_path
        self.score_model_path = score_model_path
        self.private_key = private_key
        self.device_name = device_name
        self.backend = backend

        root_core_path, _ = os.path.split(os.path.abspath(__file__))
        self.font = ImageFont.truetype(os.path.join(root_core_path, 'resource/arial.ttf'), 24)
        
        # TODO : normalization 
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
            transforms.Resize(IMAGE_SIZE if self.backend=='pytorch' else VINO_CROP_SIZE),
            transforms.ToTensor()
        ])
        
        if self.backend == 'pytorch':
            self.device = torch.device(self.device_name)

            # TODO : decrypt model with private_key

            self.detection_net = detection_model.resnet50(num_classes=NUM_CLASSES)
            self.score_net = score_model.resnet50(num_classes=1)

            self.detection_net.load_state_dict(torch.load(self.detection_model_path))
            self.score_net.load_state_dict(torch.load(self.score_model_path))

            self.detection_net.to(self.device)
            self.score_net.to(self.device)
        elif self.backend == 'openvino':
            # IR expects BGR, but our transform pipeline exports RGB
            # remember to convert model with --reverse_input_channels parameter
            
            # our normalization is implemented in transform
            # so do NOT specify --scale_values, --mean_values
            
            # after ToTensor(), we got (n, c, h, w) tensor so .numpy() should be ok
            from openvino.inference_engine import IECore
            
            self.ie = IECore()
            
            self.detection_model_bin = os.path.splitext(self.detection_model_path)[0] + '.bin'
            self.score_model_bin = os.path.splitext(self.score_model_path)[0] + '.bin'
            
            self.detection_net = self.ie.read_network(self.detection_model_path, self.detection_model_bin)
            self.score_net = self.ie.read_network(self.score_model_path, self.score_model_bin)
            
            self.detection_input_layer = next(iter(self.detection_net.inputs))
            self.detection_output_layers = sorted(iter(self.detection_net.outputs))
            self.score_input_layer = next(iter(self.score_net.inputs))
            self.score_output_layer = next(iter(self.score_net.outputs))
            
            self.detection_exec_model = self.ie.load_network(self.detection_net, det_vino_device)
            self.score_exec_models = []
            if score_vino_device == 'MULTI':
                for dev in self.ie.available_devices:
                    if 'MYRIAD' in dev:
                        self.score_exec_models.append(self.ie.load_network(self.score_net, dev))
                print('det device: {}, score MYRIAD device(s): {}'.format(det_vino_device, len(self.score_exec_models)))
            else:
                self.score_exec_model = self.ie.load_network(self.score_net, score_vino_device)
                print('det device: {}, score device: {}'.format(det_vino_device, score_vino_device))
            
            self.detection_post_processor = DetectionPostProcessor(NMS_THRESHOLD, SCORE_THRESHOLD)
        else:
            print('unknown backend {}'.format(self.backend))

    def detect(self, picture_path):
        img = Image.open(picture_path)

        if img.mode == 'I':
            img = self._convert_I16_to_L(img)

        img = img.convert('RGB')

        data = self.detect_trans(img, (torch.tensor([]), torch.tensor([]), {}))

        scale = data[1][2]["scale"]
        pad_loc = data[1][2]["pad_loc"]
        
        if self.backend == 'pytorch':
            self.detection_net.eval()
            self.detection_net.set_nms(NMS_THRESHOLD)

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
        elif self.backend == 'openvino':
            output = self.detection_exec_model.infer(inputs={self.detection_input_layer: data[0].numpy()})
            
            regression, classification = output[self.detection_output_layers[0]], output[self.detection_output_layers[1]]
            regression = torch.from_numpy(regression)
            classification = torch.from_numpy(classification)
            
            img_batch = data[0].unsqueeze(0)
            
            scores, labels, boxes = self.detection_post_processor.process(img_batch, regression, classification)
            dets = self.detection_post_processor.export(scores, labels, boxes, scale, pad_loc)
            
            return dets
        else:
            return []

    def bbox_score(self, picture_path, bboxes):
        if len(bboxes) == 0:
            return []
        
        img = Image.open(picture_path)
        
        if img.mode == 'I':
            img = self._convert_I16_to_L(img)
        
        img = img.convert('RGB')
        
        if self.backend == 'pytorch':
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
        elif self.backend == 'openvino':
            crops = [self.score_trans(img.crop(bbox)) for bbox in bboxes]
            scores = []
            
            if len(self.score_exec_models) == 0:
                for crop in crops:
                    output = self.score_exec_model.infer(inputs={self.score_input_layer: crop.numpy()})
                    score = output[self.score_output_layer][0] * STD + MEAN

                    scores.append(score)
            else:
                reqs = []
                
                for crop in crops:
                    requested = False
                    
                    while not requested:
                        for m in self.score_exec_models:
                            rid = m.get_idle_request_id()
                            if rid > -1:
                                req = m.start_async(rid, inputs={self.score_input_layer: crop.numpy()})
                                reqs.append(req)
                                
                                requested = True
                                break
                
                for req in reqs:
                    req.wait()
                    score = req.outputs[self.score_output_layer][0] * STD + MEAN
                    
                    scores.append(score)
                    
            score_bboxes = np.concatenate([
                np.array(bboxes),
                np.array(scores)
            ], axis=1)

            return score_bboxes.tolist()
        else:
            return []

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
