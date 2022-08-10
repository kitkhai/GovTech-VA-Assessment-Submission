FROM python:3.8

COPY requirements.txt .

RUN pip install -r requirements.txt

# Copy everything inside the docker image
COPY . .

EXPOSE 5000

CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
