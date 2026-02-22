FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 10000
CMD ["streamlit", "run", "app2.py", "--server.port=10000", "--server.address=0.0.0.0"]