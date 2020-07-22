import io

from minio import Minio
from minio.error import ResponseError, BucketAlreadyExists, BucketAlreadyOwnedByYou

from celery import Celery
from kombu import Queue

from config_local import *
from core import RootCore

# consts
CONFIG_FILE = 'celery-config'

DETECTION_MODEL_FILE = './data/detection_state_dict.pth' if det_model is None else det_model
SCORE_MODEL_FILE = './data/score_state_dict.pth' if score_model is None else score_model

src_bucket_name = SRC_BUCKET
dest_bucket_name = DEST_BUCKET
endpoint = 'us-east-1'

# storage
mc = Minio(MINIO_URL,
    access_key=MINIO_ACCESS,
    secret_key=MINIO_SECRET,
    secure=MINIO_SECURE)

try:
    mc.make_bucket(src_bucket_name, location=endpoint)
except BucketAlreadyOwnedByYou as err:
    pass
except BucketAlreadyExists as err:
    pass
except ResponseError as err:
    pass

try:
    mc.make_bucket(dest_bucket_name, location=endpoint)
except BucketAlreadyOwnedByYou as err:
    pass
except BucketAlreadyExists as err:
    pass
except ResponseError as err:
    pass

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
    device_name,
    backend=backend,
    det_vino_device=det_vino_device,
    score_vino_device=score_vino_device
)

# low level api
@celery.task(max_retries=3)
def detect(picture_key):
    image = mc.get_object(src_bucket_name, picture_key)
    imageArr = io.BytesIO(image.read())

    bboxes = core.detect(imageArr)

    return bboxes

@celery.task(max_retries=3)
def bbox_score(picture_key, bboxes):
    image = mc.get_object(src_bucket_name, picture_key)
    imageArr = io.BytesIO(image.read())

    score_bboxes = core.bbox_score(imageArr, bboxes)

    return score_bboxes

# high level api
@celery.task(max_retries=3)
def score(picture_key):
    image = mc.get_object(src_bucket_name, picture_key)
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
    image = mc.get_object(src_bucket_name, picture_key)
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

        mc.put_object(
            dest_bucket_name,
            result_key,
            io.BytesIO(resultArr),
            length=len(resultArr),
            content_type='image/jpeg'
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
    image = mc.get_object(src_bucket_name, picture_key)
    imageArr = io.BytesIO(image.read())

    resultArr = core.draw_score_bboxes(
        imageArr,
        score_bboxes,
        bbox_color_name=bbox_color_name,
        score_color_name=score_color_name,
        draw_score=draw_score
    )

    mc.put_object(
        dest_bucket_name,
        picture_key,
        io.BytesIO(resultArr),
        length=len(resultArr),
        content_type='image/jpeg'
    )

    return picture_key
