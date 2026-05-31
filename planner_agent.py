import os
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class PlannerAgent:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def make_plan(self, question: str, available_years=None, available_columns=None) -> Dict[str, Any]:
        available_years = available_years or ["2014", "2015", "2016", "2017", "2018"]
        available_columns = available_columns or []

        prompt = f"""
You are the planning node of an orchestrated AI analytics system.

Classify the user question and produce a JSON plan.

Available years:
{available_years}

Available dataset columns:
{available_columns}

Routes:
1. "analytics" — question requires the local financial CSV dataset.
2. "web" — question asks general background, company info, definitions, history, or external knowledge.

Analytics analysis_type options:
- sector_ranking
- company_ranking
- cross_year_comparison
- financial_strength
- undervalued_screen
- correlation
- general_profile

Return ONLY valid JSON.

JSON schema:
{{
  "route": "analytics" or "web",
  "analysis_type": "...",
  "year": "2014/2015/2016/2017/2018 or null",
  "start_year": "2014 or null",
  "end_year": "2018 or null",
  "group_by": "Sector or null",
  "metric": "Revenue/Net Income/Profit Margin/Free Cash Flow/ROE/ROIC/etc or null",
  "aggregation": "mean/sum/median/count or null",
  "sort_order": "ascending/descending or null",
  "top_n": 10,
  "filters": {{}},
  "plot_type": "bar/scatter/line/table or null",
  "needs_web": false,
  "reason": "brief explanation"
}}

Important mapping rules:
- "sales", "top line", "income from sales" usually means Revenue.
- "earnings", "profit" can mean Net Income.
- "industries" usually means Sector.
- "best", "strongest", "healthiest" usually means financial_strength descending.
- "worst", "weakest", "riskiest" usually means financial_strength ascending.
- "highest", "largest", "most" means descending.
- "lowest", "smallest", "least" means ascending.
- If the user asks what something means, route to web.
- If the user asks about NVIDIA, Apple, OPEC, oil crash, or current external facts, route to web.
- If the user asks to compare 2014 and 2018 using the dataset, route to analytics.

User question:
{question}
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You output only valid JSON. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        text = response.choices[0].message.content.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {
                "route": "web",
                "analysis_type": None,
                "year": None,
                "start_year": None,
                "end_year": None,
                "group_by": None,
                "metric": None,
                "aggregation": None,
                "sort_order": None,
                "top_n": 10,
                "filters": {},
                "plot_type": None,
                "needs_web": True,
                "reason": "Planner failed to produce valid JSON."
            }