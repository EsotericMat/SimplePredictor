FROM python:3.11-slim
LABEL authors="matanstern"
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "dataprep.py"]
CMD ["python", "pipeline.py"]
CMD ["python", "service.py"]
