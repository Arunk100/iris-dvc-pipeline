FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy model and app
COPY model.pkl /app/model.pkl
COPY app.py /app/app.py

EXPOSE 8080
ENV PYTHONUNBUFFERED=1
CMD ["python","app.py"]
