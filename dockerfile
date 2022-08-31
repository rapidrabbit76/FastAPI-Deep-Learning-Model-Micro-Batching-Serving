FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
WORKDIR /app
COPY requirements.txt  /app
RUN apt-get update  \
    && apt install -y \ 
    curl  \
    && apt-get clean 

RUN pip install --no-cache-dir -r requirements.txt 
ADD . /app

EXPOSE 8000
CMD ["uvicorn", "app_v2.main:app", "--host", "0.0.0.0"]