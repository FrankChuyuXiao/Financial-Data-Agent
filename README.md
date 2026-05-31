# FinSight Orchestrator

An orchestrated AI analytics system for ECE157C HW4.

This project supports:

- routing user questions through an orchestration node,
- answering generic-domain questions using DuckDuckGo search,
- answering analytics questions using a persistent Python analytics sandbox,
- validating analytics results with an independent validator agent,
- retrying incomplete or suspicious analysis,
- returning natural language answers and Plotly JSON visualizations,
- supporting a simple chatbot frontend.

## Dataset

Place at least two yearly CSV files from the Kaggle dataset:

`200+ Financial Indicators of US Stocks (2014–2018)`

inside:

```text
datasets/
```

Recommended:

```text
datasets/2014_Financial_Data.csv
datasets/2018_Financial_Data.csv
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add your OpenAI API key.

## Run Backend

```bash
python -m uvicorn backend.main:app --reload
```

Then open a new terminal and type:

```text
cd C:\Chuyu Xiao\UCSB_undergrad\ECE157C\ECE157C_HW4\project\project
python -m http.server 5500
```

Open:

http://localhost:5500/frontend/index.html

in your browser.

## Example Questions

Generic-domain:

```text
What is EBITDA and why is it useful?
What caused the 2016 oil-price crash?
```

Analytics:

```text
Which sectors had the highest average revenue in 2018?
Compare revenue and net income by sector between 2014 and 2018.
Find undervalued companies with positive free cash flow and low debt.
Which sectors look financially strongest from 2014 to 2018?
```

## Main Files

```text
orchestrator.py        routes questions
analytics_agent.py     iterative analytics/code execution agent
validator_agent.py     checks correctness/completeness
web_search.py          DuckDuckGo search tool
memory.py              simple conversation memory
backend/main.py        FastAPI API
frontend/index.html    chatbot UI
```