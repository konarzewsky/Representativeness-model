FROM python:3.11.8-slim

RUN pip install --upgrade pip wheel
COPY ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY . /app
WORKDIR /app
RUN pip install --user -e .

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "5000", "--reload"]
