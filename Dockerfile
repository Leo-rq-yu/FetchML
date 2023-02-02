FROM python:3.8.10
ADD requirements.txt /
RUN pip install -r /requirements.txt
ADD fetchmloa.py /
ENV PYTHONUNBUFFERED=1
CMD ["python","./fetchmloa.py"]