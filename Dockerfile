FROM python:3.11-slim

WORKDIR /code

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir gradio tokenizers

COPY . .

EXPOSE 7860

ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
