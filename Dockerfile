FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies for scapy/pyshark/tshark
RUN apt-get update && apt-get install -y --no-install-recommends \
    tshark libpcap-dev build-essential ca-certificates git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Initialize DB
RUN python -c "from quantum_anomaly.storage.db import init_db; init_db()"

EXPOSE 8501

# CAPs for sniffing are up to the runtime flags
CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
