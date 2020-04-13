# root-server

## deployment

1. Put detection_state_dict.pth and score_state_dict.pth to data/. You may need to re-save state_dict with extract_state_dict.ipynb.  
2. Prepare celery-config.py and config.py.  

celery-config.py:  
```python
broker_url = 'redis://:[REDIS_PASSWORD]@r-wz9j4vwobcb81ws9mapd.redis.rds.aliyuncs.com:6379/0'
result_backend = 'redis://:[REDIS_PASSWORD]@r-wz9j4vwobcb81ws9mapd.redis.rds.aliyuncs.com:6379/0'
```
config.py:
```python
access_key_id = 'xxxxx'
access_key_secret = 'yyyyy'
```
3. start celery worker:
```
celery -A root-worker worker
```
or with nohup:  
```
nohup celery -A root-worker worker > worker.log &
```
