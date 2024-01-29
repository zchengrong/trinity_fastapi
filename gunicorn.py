daemon = True  # 是否守护
chdir = '.'  # 项目地址
worker_class = 'uvicorn.workers.UvicornWorker'
workers = 10
threads = 10
loglevel = 'debug'  # 日志级别
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
accesslog = "gunicorn_access.log"
errorlog = "gunicorn_error.log"


# gunicorn main:app -b 0.0.0.0:9000  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9001  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9002  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9003  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9004  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9005  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9006  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9007  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9008  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9009  -c gunicorn.py
# gunicorn main:app -b 0.0.0.0:9010  -c gunicorn.py
