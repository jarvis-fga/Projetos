FROM python:3 
ENV PYTHONUNBUFFERED 1 

RUN mkdir /code 
WORKDIR /code 

ADD . /code/
RUN pip install -r requirements.txt

RUN chmod +x boot.sh

EXPOSE 8888
CMD ["/bin/sh", "./boot.sh"]
