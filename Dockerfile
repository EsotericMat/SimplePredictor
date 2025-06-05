FROM python:3.11-slim
LABEL authors="matanstern"
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["sh", "-c", "python dataprep.py && python pipeline.py && python service.py"]
