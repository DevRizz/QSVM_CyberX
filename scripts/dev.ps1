Param()
$ErrorActionPreference = "Stop"

py -3 -m venv .venv
. .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -c "from quantum_anomaly.storage.db import init_db; init_db()"
streamlit run streamlit_app/app.py
