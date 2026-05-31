from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import traceback
import re

import pandas as pd
import numpy as np
import plotly.express as px


class PersistentPythonSandbox:
    def __init__(self):
        self.namespace: Dict[str, Any] = {
            "pd": pd,
            "np": np,
            "px": px,
            "Path": Path,
            "json": json,
        }

    def run(self, code: str) -> Dict[str, Any]:
        out = {"success": True, "error": None}
        try:
            exec(code, self.namespace)
        except Exception:
            out["success"] = False
            out["error"] = traceback.format_exc()
        return out


class AnalyticsAgent:
    def __init__(self, dataset_dir: str = "datasets"):
        self.dataset_dir = Path(dataset_dir)
        self.sandbox = PersistentPythonSandbox()

    # -----------------------------
    # File / column helpers
    # -----------------------------

    def _find_csv_files(self) -> List[Path]:
        return sorted(self.dataset_dir.glob("*.csv"))

    def _extract_year(self, text: str) -> Optional[str]:
        for year in ["2014", "2015", "2016", "2017", "2018"]:
            if year in str(text):
                return year
        return None

    def _load_dataframes(self) -> Dict[str, pd.DataFrame]:
        dataframes = {}
        for path in self._find_csv_files():
            year = self._extract_year(path.name)
            key = year if year else path.stem
            try:
                dataframes[key] = pd.read_csv(path)
            except Exception as e:
                print(f"Could not read {path}: {e}")
        return dataframes

    def _clean_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        for col in cleaned.columns:
            if col.lower() == "sector":
                continue
            converted = pd.to_numeric(cleaned[col], errors="coerce")
            if converted.notna().sum() > 0:
                cleaned[col] = converted
        return cleaned

    def _safe_col(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        lower_map = {c.lower(): c for c in df.columns}

        for candidate in candidates:
            if candidate.lower() in lower_map:
                return lower_map[candidate.lower()]

        # fuzzy contains match
        for candidate in candidates:
            c = candidate.lower()
            for real_lower, real_col in lower_map.items():
                if c in real_lower or real_lower in c:
                    return real_col

        return None

    def get_available_years(self) -> List[str]:
        years = []
        for p in self._find_csv_files():
            y = self._extract_year(p.name)
            if y:
                years.append(y)
        return sorted(set(years))

    def get_available_columns(self) -> List[str]:
        dfs = self._load_dataframes()
        if not dfs:
            return []
        return list(next(iter(dfs.values())).columns)

    def _choose_year(self, question: str, dataframes: Dict[str, pd.DataFrame], plan: Optional[Dict[str, Any]] = None) -> str:
        if plan and plan.get("year") and str(plan["year"]) in dataframes:
            return str(plan["year"])

        y = self._extract_year(question)
        if y and y in dataframes:
            return y

        for y in ["2018", "2017", "2016", "2015", "2014"]:
            if y in dataframes:
                return y

        return sorted(dataframes.keys())[-1]

    def _metric_from_text(self, question: str, df: pd.DataFrame, plan: Optional[Dict[str, Any]] = None) -> str:
        if plan and plan.get("metric"):
            col = self._safe_col(df, [str(plan["metric"])])
            if col:
                return col

        q = question.lower()

        mapping = [
            (["sales", "top line", "revenue"], ["Revenue"]),
            (["net income", "earnings", "profit"], ["Net Income"]),
            (["gross profit"], ["Gross Profit"]),
            (["gross margin"], ["Gross Margin", "grossProfitMargin"]),
            (["profit margin", "net margin"], ["Profit Margin", "netProfitMargin", "Net Profit Margin"]),
            (["free cash flow", "fcf"], ["Free Cash Flow"]),
            (["operating cash flow"], ["Operating Cash Flow"]),
            (["roe", "return on equity"], ["ROE", "returnOnEquity"]),
            (["roic", "return on invested capital", "capital employed"], ["ROIC", "returnOnCapitalEmployed"]),
            (["debt to equity", "leverage"], ["Debt to Equity", "debtEquityRatio"]),
            (["current ratio", "liquidity"], ["Current ratio", "currentRatio"]),
            (["market cap", "market capitalization"], ["Market Cap"]),
            (["enterprise value"], ["Enterprise Value"]),
            (["pe", "p/e"], ["PE ratio", "priceEarningsRatio"]),
            (["price to sales", "p/s"], ["Price to Sales Ratio", "priceToSalesRatio"]),
        ]

        for phrases, cols in mapping:
            if any(p in q for p in phrases):
                col = self._safe_col(df, cols)
                if col:
                    return col

        return self._safe_col(df, ["Revenue"]) or df.select_dtypes(include=[np.number]).columns[0]

    def _sort_ascending(self, question: str, plan: Optional[Dict[str, Any]] = None) -> bool:
        if plan and plan.get("sort_order"):
            return str(plan["sort_order"]).lower() == "ascending"

        q = question.lower()
        low_words = ["lowest", "least", "smallest", "weakest", "worst", "bottom", "cheapest", "low"]
        high_words = ["highest", "most", "largest", "strongest", "best", "top", "high"]

        if any(w in q for w in low_words):
            return True
        if any(w in q for w in high_words):
            return False
        return False

    def _top_n(self, plan: Optional[Dict[str, Any]] = None) -> int:
        if plan and plan.get("top_n"):
            try:
                return int(plan["top_n"])
            except Exception:
                pass
        return 10

    # -----------------------------
    # Main entry points
    # -----------------------------

    def answer(self, question: str, retry_feedback: Dict[str, Any] | None = None) -> Dict[str, Any]:
        return self.answer_with_plan(question, plan=None)

    def answer_with_plan(self, question: str, plan: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        trace = []
        plots = []

        dataframes = self._load_dataframes()
        if not dataframes:
            return {
                "route": "analytics",
                "success": False,
                "final_answer": "No CSV files were found in the datasets folder.",
                "plots": [],
                "trace": [],
                "used_files": []
            }

        for year, df in dataframes.items():
            dataframes[year] = self._clean_numeric(df)

        used_files = [p.name for p in self._find_csv_files()]

        trace.append({
            "iteration": 1,
            "action": "inspect_datasets",
            "observation": {
                "files": used_files,
                "tables": {
                    y: {"rows": len(df), "cols": len(df.columns)}
                    for y, df in dataframes.items()
                }
            }
        })

        analysis_type = None
        if plan:
            analysis_type = plan.get("analysis_type")

        if not analysis_type:
            analysis_type = self._infer_analysis_type(question)

        if analysis_type == "sector_ranking":
            result = self._sector_ranking(question, dataframes, trace, plots, plan)
        elif analysis_type == "company_ranking":
            result = self._company_ranking(question, dataframes, trace, plots, plan)
        elif analysis_type == "cross_year_comparison":
            result = self._cross_year_comparison(question, dataframes, trace, plots, plan)
        elif analysis_type == "financial_strength":
            result = self._financial_strength(question, dataframes, trace, plots)
        elif analysis_type == "undervalued_screen":
            result = self._undervalued_screen(question, dataframes, trace, plots, plan)
        elif analysis_type == "correlation":
            result = self._correlation(question, dataframes, trace, plots, plan)
        else:
            result = self._general_profile(question, dataframes, trace, plots, plan)

        result["used_files"] = used_files
        result["route"] = "analytics"
        result["success"] = True
        return result

    def _infer_analysis_type(self, question: str) -> str:
        q = question.lower()

        if any(x in q for x in ["relationship", "correlation", "associated", "tend to"]):
            return "correlation"

        if any(x in q for x in ["undervalued", "cheap", "low valuation", "bargain"]):
            return "undervalued_screen"

        if any(x in q for x in ["financially strongest", "financially weakest", "healthiest", "riskiest", "strong balance sheet", "weak balance sheet"]):
            return "financial_strength"

        if ("2014" in q and "2018" in q) or "from 2014 to 2018" in q or "between 2014 and 2018" in q or "cross-year" in q:
            return "cross_year_comparison"

        if any(x in q for x in ["company", "companies", "stocks"]):
            return "company_ranking"

        if any(x in q for x in ["sector", "sectors", "industry", "industries"]):
            return "sector_ranking"

        return "general_profile"

    # -----------------------------
    # Analysis implementations
    # -----------------------------

    def _sector_ranking(self, question, dataframes, trace, plots, plan=None):
        year = self._choose_year(question, dataframes, plan)
        df = dataframes[year]

        sector_col = self._safe_col(df, ["Sector"])
        metric_col = self._metric_from_text(question, df, plan)
        ascending = self._sort_ascending(question, plan)
        top_n = self._top_n(plan)

        if not sector_col or not metric_col:
            return {"final_answer": "Could not find required sector or metric column.", "plots": plots, "trace": trace}

        aggregation = (plan or {}).get("aggregation", "mean")
        if aggregation == "sum":
            grouped = df.groupby(sector_col)[metric_col].sum(numeric_only=True).reset_index()
        elif aggregation == "median":
            grouped = df.groupby(sector_col)[metric_col].median(numeric_only=True).reset_index()
        else:
            grouped = df.groupby(sector_col)[metric_col].mean(numeric_only=True).reset_index()
            aggregation = "mean"

        grouped = grouped.sort_values(metric_col, ascending=ascending).head(top_n)

        direction = "lowest" if ascending else "highest"

        trace.append({
            "iteration": 2,
            "action": "sector_ranking",
            "observation": {
                "year": year,
                "group_by": sector_col,
                "metric": metric_col,
                "aggregation": aggregation,
                "direction": direction
            }
        })

        fig = px.bar(
            grouped,
            x=sector_col,
            y=metric_col,
            title=f"Sectors with the {direction.title()} {aggregation.title()} {metric_col} in {year}"
        )

        plots.append({"title": fig.layout.title.text, "plotly_json": fig.to_json()})

        trace.append({
            "iteration": 3,
            "action": "generate_plot",
            "observation": "Generated sector ranking bar chart."
        })

        final_answer = (
            f"For {year}, I ranked sectors by {aggregation} {metric_col} in {direction} order. "
            f"The first sector in this ranking is {grouped.iloc[0][sector_col]}, with a value of "
            f"{grouped.iloc[0][metric_col]:,.2f}.\n\n"
            f"Results:\n{grouped.to_string(index=False)}"
        )

        trace.append({
            "iteration": 4,
            "action": "finalize_answer",
            "observation": "Prepared sector ranking answer."
        })

        return {"final_answer": final_answer, "plots": plots, "trace": trace}

    def _company_ranking(self, question, dataframes, trace, plots, plan=None):
        year = self._choose_year(question, dataframes, plan)
        df = dataframes[year]

        metric_col = self._metric_from_text(question, df, plan)
        ascending = self._sort_ascending(question, plan)
        top_n = self._top_n(plan)

        name_col = df.columns[0]
        sector_col = self._safe_col(df, ["Sector"])

        cols = [name_col, metric_col]
        if sector_col:
            cols.append(sector_col)

        ranked = df[cols].dropna(subset=[metric_col]).sort_values(metric_col, ascending=ascending).head(top_n)

        direction = "lowest" if ascending else "highest"

        trace.append({
            "iteration": 2,
            "action": "company_ranking",
            "observation": {
                "year": year,
                "metric": metric_col,
                "direction": direction
            }
        })

        fig = px.bar(
            ranked,
            x=name_col,
            y=metric_col,
            color=sector_col if sector_col else None,
            title=f"Companies with the {direction.title()} {metric_col} in {year}"
        )
        plots.append({"title": fig.layout.title.text, "plotly_json": fig.to_json()})

        final_answer = (
            f"For {year}, I ranked companies by {metric_col} in {direction} order. "
            f"The first company in this ranking is {ranked.iloc[0][name_col]}, with a value of "
            f"{ranked.iloc[0][metric_col]:,.2f}.\n\n"
            f"Results:\n{ranked.to_string(index=False)}"
        )

        trace.append({
            "iteration": 3,
            "action": "finalize_answer",
            "observation": "Prepared company ranking answer."
        })

        return {"final_answer": final_answer, "plots": plots, "trace": trace}

    def _cross_year_comparison(self, question, dataframes, trace, plots, plan=None):
        start_year = str((plan or {}).get("start_year") or "2014")
        end_year = str((plan or {}).get("end_year") or "2018")

        if start_year not in dataframes or end_year not in dataframes:
            return {
                "final_answer": f"Missing required files for {start_year} and {end_year}.",
                "plots": plots,
                "trace": trace
            }

        df_start = dataframes[start_year]
        df_end = dataframes[end_year]

        sector_col = self._safe_col(df_end, ["Sector"])
        metric_col = self._metric_from_text(question, df_end, plan)

        if not sector_col or not metric_col:
            return {"final_answer": "Could not find required columns for cross-year comparison.", "plots": plots, "trace": trace}

        s1 = df_start.groupby(sector_col)[metric_col].mean(numeric_only=True).reset_index()
        s2 = df_end.groupby(sector_col)[metric_col].mean(numeric_only=True).reset_index()

        merged = s1.merge(s2, on=sector_col, suffixes=(f"_{start_year}", f"_{end_year}"))
        old_col = f"{metric_col}_{start_year}"
        new_col = f"{metric_col}_{end_year}"

        merged["Change"] = merged[new_col] - merged[old_col]
        merged["Growth %"] = merged["Change"] / merged[old_col].replace(0, np.nan) * 100

        ascending = self._sort_ascending(question, plan)
        merged = merged.sort_values("Growth %", ascending=ascending).head(self._top_n(plan))

        direction = "lowest" if ascending else "highest"

        trace.append({
            "iteration": 2,
            "action": "cross_year_comparison",
            "observation": {
                "start_year": start_year,
                "end_year": end_year,
                "metric": metric_col,
                "direction": direction
            }
        })

        fig = px.bar(
            merged,
            x=sector_col,
            y="Growth %",
            title=f"{metric_col} Growth by Sector from {start_year} to {end_year}"
        )
        plots.append({"title": fig.layout.title.text, "plotly_json": fig.to_json()})

        final_answer = (
            f"I compared average sector {metric_col} from {start_year} to {end_year}. "
            f"The sector with the {direction} growth in this ranking is {merged.iloc[0][sector_col]}, "
            f"with growth of {merged.iloc[0]['Growth %']:.2f}%.\n\n"
            f"Results:\n{merged.to_string(index=False)}"
        )

        trace.append({
            "iteration": 3,
            "action": "finalize_answer",
            "observation": "Prepared cross-year comparison answer."
        })

        return {"final_answer": final_answer, "plots": plots, "trace": trace}

    def _financial_strength(self, question, dataframes, trace, plots):
        weakest = any(w in question.lower() for w in ["weakest", "weak", "worst", "riskiest"])

        rows = []
        for year, df in dataframes.items():
            sector_col = self._safe_col(df, ["Sector"])
            if not sector_col:
                continue

            metric_candidates = [
                self._safe_col(df, ["Revenue"]),
                self._safe_col(df, ["Net Income"]),
                self._safe_col(df, ["Free Cash Flow"]),
                self._safe_col(df, ["ROE", "returnOnEquity"]),
                self._safe_col(df, ["ROIC", "returnOnCapitalEmployed"]),
                self._safe_col(df, ["Current ratio", "currentRatio"]),
                self._safe_col(df, ["Debt to Equity", "debtEquityRatio"]),
            ]
            metrics = [m for m in metric_candidates if m]
            g = df.groupby(sector_col)[metrics].mean(numeric_only=True).reset_index()
            g["Year"] = year
            rows.append(g)

        combined = pd.concat(rows, ignore_index=True)
        sector_avg = combined.groupby("Sector").mean(numeric_only=True).reset_index()

        positive_names = ["Revenue", "Net Income", "Free Cash Flow", "ROE", "returnOnEquity", "ROIC", "returnOnCapitalEmployed", "Current ratio", "currentRatio"]
        negative_names = ["Debt to Equity", "debtEquityRatio"]

        score_parts = []
        for col in sector_avg.columns:
            if col == "Sector":
                continue
            if col in positive_names:
                score_parts.append(sector_avg[col].rank(pct=True))
            elif col in negative_names:
                score_parts.append(1 - sector_avg[col].rank(pct=True))

        sector_avg["Financial Strength Score"] = sum(score_parts) / len(score_parts)
        sector_avg = sector_avg.sort_values("Financial Strength Score", ascending=weakest).head(10)

        direction = "weakest" if weakest else "strongest"

        trace.append({
            "iteration": 2,
            "action": "financial_strength_score",
            "observation": f"Computed {direction} sector ranking using multi-metric financial strength score."
        })

        fig = px.bar(
            sector_avg,
            x="Sector",
            y="Financial Strength Score",
            title=f"Financially {direction.title()} Sectors from 2014 to 2018"
        )
        plots.append({"title": fig.layout.title.text, "plotly_json": fig.to_json()})

        final_answer = (
            f"The financially {direction} sectors from 2014 to 2018 were ranked using a composite score. "
            "The score rewards stronger revenue, net income, free cash flow, ROE, ROIC, and liquidity, "
            "and penalizes higher debt-to-equity.\n\n"
            f"Results:\n{sector_avg.to_string(index=False)}"
        )

        return {"final_answer": final_answer, "plots": plots, "trace": trace}

    def _undervalued_screen(self, question, dataframes, trace, plots, plan=None):
        year = self._choose_year(question, dataframes, plan)
        df = dataframes[year].copy()

        sector_col = self._safe_col(df, ["Sector"])
        pe = self._safe_col(df, ["PE ratio", "priceEarningsRatio"])
        pb = self._safe_col(df, ["PB ratio", "priceBookValueRatio", "priceToBookRatio"])
        fcf = self._safe_col(df, ["Free Cash Flow"])
        debt = self._safe_col(df, ["Debt to Equity", "debtEquityRatio"])
        roe = self._safe_col(df, ["ROE", "returnOnEquity"])
        ni = self._safe_col(df, ["Net Income"])

        criteria = pd.Series(True, index=df.index)
        if pe:
            criteria &= df[pe].between(0, df[pe].quantile(0.35))
        if pb:
            criteria &= df[pb].between(0, df[pb].quantile(0.50))
        if fcf:
            criteria &= df[fcf] > 0
        if debt:
            criteria &= df[debt] < df[debt].quantile(0.65)
        if roe:
            criteria &= df[roe] > df[roe].median()
        if ni:
            criteria &= df[ni] > 0

        candidates = df[criteria].copy()

        score_cols = [c for c in [pe, pb, debt] if c]
        if score_cols and len(candidates) > 0:
            candidates["Value Score"] = candidates[score_cols].rank(pct=True).mean(axis=1)
            candidates = candidates.sort_values("Value Score").head(10)

        display = [df.columns[0]] + [c for c in [sector_col, pe, pb, fcf, debt, roe, ni] if c]
        if "Value Score" in candidates:
            display.append("Value Score")

        trace.append({
            "iteration": 2,
            "action": "undervalued_screen",
            "observation": f"Found {len(candidates)} candidates in {year}."
        })

        if len(candidates) > 0 and pe and roe:
            fig = px.scatter(
                candidates,
                x=pe,
                y=roe,
                color=sector_col if sector_col else None,
                title=f"Undervalued Candidates in {year}"
            )
            plots.append({"title": fig.layout.title.text, "plotly_json": fig.to_json()})

        final_answer = (
            f"In {year}, I screened for undervalued companies using low valuation ratios, positive free cash flow, "
            "manageable debt, positive net income, and stronger ROE.\n\n"
            f"Results:\n{candidates[display].to_string(index=False) if len(candidates) else 'No candidates matched all filters.'}"
        )

        return {"final_answer": final_answer, "plots": plots, "trace": trace}

    def _correlation(self, question, dataframes, trace, plots, plan=None):
        year = self._choose_year(question, dataframes, plan)
        df = dataframes[year]

        q = question.lower()

        if "r&d" in q or "research" in q:
            x = self._safe_col(df, ["R&D to Revenue", "R&D Expenses"])
            y = self._safe_col(df, ["Revenue Growth"])
        elif "roic" in q:
            x = self._safe_col(df, ["ROIC", "returnOnCapitalEmployed"])
            y = self._safe_col(df, ["2015 PRICE VAR [%]", "Class"])
        else:
            nums = df.select_dtypes(include=[np.number]).columns.tolist()
            x, y = nums[0], nums[1]

        sector_col = self._safe_col(df, ["Sector"])

        work = df[[x, y] + ([sector_col] if sector_col else [])].dropna()
        corr = work[x].corr(work[y])

        trace.append({
            "iteration": 2,
            "action": "correlation",
            "observation": f"Computed correlation between {x} and {y}: {corr:.3f}"
        })

        fig = px.scatter(
            work,
            x=x,
            y=y,
            color=sector_col if sector_col else None,
            title=f"Relationship between {x} and {y} in {year}"
        )
        plots.append({"title": fig.layout.title.text, "plotly_json": fig.to_json()})

        strength = "strong" if abs(corr) >= 0.7 else "moderate" if abs(corr) >= 0.3 else "weak"
        direction = "positive" if corr > 0 else "negative"

        final_answer = (
            f"In {year}, the correlation between {x} and {y} is {corr:.3f}. "
            f"This suggests a {strength} {direction} relationship."
        )

        return {"final_answer": final_answer, "plots": plots, "trace": trace}

    def _general_profile(self, question, dataframes, trace, plots, plan=None):
        year = self._choose_year(question, dataframes, plan)
        df = dataframes[year]

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        summary = df[numeric_cols[:10]].describe().T

        trace.append({
            "iteration": 2,
            "action": "general_profile",
            "observation": f"Generated descriptive profile for {year}."
        })

        final_answer = (
            f"I inspected the {year} financial dataset. It contains {len(df)} rows and {len(df.columns)} columns.\n\n"
            f"Summary statistics:\n{summary.to_string()}"
        )

        return {"final_answer": final_answer, "plots": plots, "trace": trace}