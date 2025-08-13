#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -c "from quantum_anomaly.storage.db import init_db; init_db()"
streamlit run streamlit_app/app.py
