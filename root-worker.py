import io

import oss2
from celery import Celery
from kombu import Queue

from config import *
from core import RootCore

# consts
CONFIG_FILE = 'celery-config'

DETECTION_MODEL_FILE = './data/detection_state_dict.pth'
SCORE_MODEL_FILE = './data/score_state_dict.pth'
DEVICE_NAME = 'cpu'

src_bucket_name = 'root-ai-src'
dest_bucket_name = 'root-ai-dest'
endpoint = 'oss-cn-chengdu.aliyuncs.com'

src_bucket = oss2.Bucket(
    oss2.Auth(access_key_id, access_key_secret),
    endpoint,
    src_bucket_name
)
dest_bucket = oss2.Bucket(
    oss2.Auth(access_key_id, access_key_secret),
    endpoint,
    dest_bucket_name
)

# read config & init objects
celery = Celery()
celery.config_from_object(CONFIG_FILE)
celery.conf.task_queues = (
     Queue('task', routing_key='task'),
)
core = RootCore.RootCore(
    DETECTION_MODEL_FILE,
    SCORE_MODEL_FILE,
    '',
    DEVICE_NAME
)

# low level api
@celery.task(max_retries=3)
def detect(picture_key):
    image = src_bucket.get_object(picture_key)
    imageArr = io.BytesIO(image.read())

    bboxes = core.detect(imageArr)

    return bboxes

@celery.task(max_retries=3)
def bbox_score(picture_key, bboxes):
    image = src_bucket.get_object(picture_key)
    imageArr = io.BytesIO(image.read())

    score_bboxes = core.bbox_score(imageArr, bboxes)

    return score_bboxes

# high level api
@celery.task(max_retries=3)
def score(picture_key):
    image = src_bucket.get_object(picture_key)
    imageArr = io.BytesIO(image.read())

    all_in_one_score_bboxes = core.score(imageArr)

    return all_in_one_score_bboxes

@celery.task(max_retries=3)
def score_and_draw(
    picture_key,
    result_pattern,
    bbox_color_name=RootCore.BBOX_COLOR_NAME,
    score_color_name=RootCore.SCORE_COLOR_NAME,
    bbox_line_width=RootCore.BBOX_LINE_WIDTH,
    draw_score=True
):
    image = src_bucket.get_object(picture_key)
    imageArr = io.BytesIO(image.read())

    all_in_one_score_bboxes = core.score(imageArr)

    for i, sbbox in enumerate(all_in_one_score_bboxes):
        resultArr = core.draw_score_bboxes(
            imageArr,
            [sbbox],
            bbox_color_name=bbox_color_name,
            score_color_name=score_color_name,
            draw_score=draw_score
        )

        result_key = result_pattern.format(i)

        dest_bucket.put_object(
            result_key,
            resultArr,
            headers={'content-length': str(len(resultArr))}
        )

        all_in_one_score_bboxes[i].append(result_key)

    return all_in_one_score_bboxes

# util api
@celery.task(max_retires=3)
def draw_score_bboxes(
    picture_key,
    score_bboxes,
    bbox_color_name=RootCore.BBOX_COLOR_NAME,
    score_color_name=RootCore.SCORE_COLOR_NAME,
    bbox_line_width=RootCore.BBOX_LINE_WIDTH,
    draw_score=True
):
    image = src_bucket.get_object(picture_key)
    imageArr = io.BytesIO(image.read())

    resultArr = core.draw_score_bboxes(
        imageArr,
        score_bboxes,
        bbox_color_name=bbox_color_name,
        score_color_name=score_color_name,
        draw_score=draw_score
    )

    dest_bucket.put_object(picture_key, resultArr, headers={'content-length': str(len(resultArr))})

    return picture_key
