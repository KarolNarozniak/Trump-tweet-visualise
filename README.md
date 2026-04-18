# Trump Tweet Visualize - Layer 1

Weekly mention-network pipeline and web app for `tweets_01-08-2021.csv`.

## Setup

```bash
python -m venv venv
venv\Scripts\python.exe -m ensurepip --upgrade
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe -m pip install -e .
```

```bash
venv\Scripts\activate
```

## Build Artifacts

```bash
venv\Scripts\python.exe -m trump_graph build --input tweets_01-08-2021.csv --out data/processed --min-mention-count 1 --include-retweets
```

## Run App

```bash
venv\Scripts\python.exe -m streamlit run app/main.py
```

## Run Tests

```bash
venv\Scripts\python.exe -m pytest -q
```
