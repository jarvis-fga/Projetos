FROM python:3 
ENV PYTHONUNBUFFERED 1 

RUN mkdir /code 
WORKDIR /code 

ADD . /code/
RUN pip install scikit-learn pandas matplotlib scipy jupyter nltk

RUN chmod +x boot.sh

EXPOSE 8888
CMD ["/bin/sh", "./boot.sh"]