# root-server

## deployment

0. To kill celery task workers, use:  
    ```
    ps -ef | grep [YOUR_WORKER_NAME] | awk '{ print $2 }' | xargs kill -9
    ```
    Do NOT ```grep celery``` or you may kill other tasks' workers.  

1. Put detection_state_dict.pth and score_state_dict.pth to data/.  
    If model was trained with DataParallel, you may need to re-save state_dict with ```extract_state_dict.ipynb```.  

2. Prepare celery-config.py and config.py.  
    celery-config.py:  
    ```python
    broker_url = 'redis://:[REDIS_PASSWORD]@r-wz9j4vwobcb81ws9mapd.redis.rds.aliyuncs.com:6379/0'
    result_backend = 'redis://:[REDIS_PASSWORD]@r-wz9j4vwobcb81ws9mapd.redis.rds.aliyuncs.com:6379/0'
    worker_pool = 'solo' # required for pytorch non-forkable lib
    task_time_limit = 60 # hard time limit to prevent hanging
    ```

    config.py:
    ```python
    access_key_id = 'xxxxx'
    access_key_secret = 'yyyyy'
    ```

3. start celery worker:
    ```
    celery -A root-worker worker --concurrency 1
    ```
    or with nohup:  
    ```
    nohup celery -A root-worker worker --concurrency 1 > worker.log &
    ```
