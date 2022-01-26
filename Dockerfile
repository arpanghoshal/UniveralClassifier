FROM tensorflow/tensorflow:latest

ADD . /python-flask
WORKDIR /python-flask

RUN pip install -r requirements.txt

EXPOSE 9000

CMD [ "python3", "main.py"]