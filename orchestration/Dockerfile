FROM apache/airflow:2.10.5-python3.11

USER root
# playwright needs these system libs for headless chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 libcups2 libdrm2 \
    libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 \
    libgbm1 libasound2 libpango-1.0-0 libcairo2 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /tmp/requirements.txt
# cpu-only torch first -- avoids pulling several GB of unneeded CUDA wheels
RUN pip install --no-cache-dir --timeout 300 --retries 5 torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --timeout 300 --retries 5 -r /tmp/requirements.txt
RUN python -m playwright install chromium

ENV PYTHONPATH=/opt/project
