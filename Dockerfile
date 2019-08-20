FROM python:3.7.2
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r ./requirements.txt


COPY latePaymentsModel.pkl /app
COPY model_columns.pkl /app
COPY application.py /app

EXPOSE 9999:9999
ENTRYPOINT python application.py 9999
#CMD ["python", "application.py"]~
