FROM python:3.12-slim-bookworm

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY handler.py .

# The weights stay on the network volume (see MODEL_PATH in handler.py), not in the image.
CMD ["python", "-u", "handler.py"]
