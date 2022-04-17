FROM ubuntu:20.04

RUN apt-get update && apt-get install python3 python3-pip git -y

COPY . /Docker

WORKDIR /Docker

RUN pip3 install -r requirements.txt

RUN pip3 install tensorflow --no-cache-dir

CMD ["python3", "src/main.py"]