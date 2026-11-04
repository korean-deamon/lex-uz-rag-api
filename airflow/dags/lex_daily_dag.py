"""
Airflow DAG: once a day, check lex.uz for new/updated documents.
If anything new or changed is found, it gets chunked, embedded, and pushed into Qdrant
so the chatbot's answers stay current.

Drop this file into your Airflow `dags/` folder. Assumes this project's root (with
scraper/, ingest/, backend/) is on PYTHONPATH / installed as a package inside the
Airflow worker environment, and that .env (OPENAI_API_KEY, QDRANT_HOST, etc) is loaded
there too.
"""
from __future__ import annotations

import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def _scrape(**context):
    from scraper.scraper import run as scrape_run
    changed_docs = scrape_run()
    # hand the changed docs to the next task via XCom
    context["ti"].xcom_push(key="changed_docs", value=changed_docs)
    print(f"[dag] scrape found {len(changed_docs)} new/changed docs")


def _ingest(**context):
    changed_docs = context["ti"].xcom_pull(key="changed_docs", task_ids="scrape_lex_uz")
    if not changed_docs:
        print("[dag] nothing new today, skipping ingest")
        return
    from ingest.ingest import ingest_docs
    ingest_docs(changed_docs)


default_args = {
    "owner": "lex-rag-bot",
    "retries": 2,
    "retry_delay": datetime.timedelta(minutes=10),
}

with DAG(
    dag_id="lex_uz_daily_sync",
    description="Daily check of lex.uz for new/updated docs -> embed -> Qdrant",
    schedule="@daily",
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["lex-uz", "rag", "scraper"],
) as dag:

    scrape_task = PythonOperator(
        task_id="scrape_lex_uz",
        python_callable=_scrape,
    )

    ingest_task = PythonOperator(
        task_id="ingest_new_docs",
        python_callable=_ingest,
    )

    scrape_task >> ingest_task
