FROM python:3.5

EXPOSE 8081

RUN apt-get update

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app


COPY requirements.txt /usr/src/app/
RUN pip install -r requirements.txt

COPY . /usr/src/app/

CMD python src/run.py