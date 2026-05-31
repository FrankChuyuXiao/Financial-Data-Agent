from typing import Dict, Any, List


class ValidatorAgent:
    """
    Independent validator/checker agent.

    It does more than check whether code executed.
    It checks:
    - whether the correct route was used,
    - whether the correct year was used,
    - whether the correct metric was used,
    - whether cross-year questions used multiple yearly files,
    - whether strongest/weakest/highest/lowest direction matches the answer,
    - whether visualization was generated when needed,
    - whether the answer is complete enough.
    """

    def validate(self, question: str, analytics_result: Dict[str, Any]) -> Dict[str, Any]:
        q = question.lower()

        answer = analytics_result.get("final_answer", "")
        answer_lower = answer.lower()

        trace = analytics_result.get("trace", [])
        plots = analytics_result.get("plots", [])
        used_files = analytics_result.get("used_files", [])
        plan = analytics_result.get("plan", {})

        issues: List[str] = []
        suggestions: List[str] = []
        checks_passed: List[str] = []

        # -----------------------------
        # 1. Basic execution check
        # -----------------------------
        if not analytics_result.get("success", False):
            issues.append("The analytics agent did not successfully complete the analysis.")
            suggestions.append("Retry the analysis after simplifying the plan and confirming available columns.")
        else:
            checks_passed.append("The analytics agent completed execution successfully.")

        # -----------------------------
        # 2. Answer completeness
        # -----------------------------
        if len(answer.strip()) < 120:
            issues.append("The final answer is too short and may not sufficiently explain the result.")
            suggestions.append("Add a clear interpretation, metric name, year, and top results.")

        if "results:" not in answer_lower and "summary" not in answer_lower:
            issues.append("The answer does not appear to include a result table or ranked summary.")
            suggestions.append("Include a table or concise ranked summary in the final answer.")
        else:
            checks_passed.append("The answer includes a result summary or table.")

        # -----------------------------
        # 3. Year validation
        # -----------------------------
        requested_years = [year for year in ["2014", "2015", "2016", "2017", "2018"] if year in q]

        for year in requested_years:
            if year not in answer_lower:
                issues.append(f"The question requested year {year}, but the answer does not clearly mention {year}.")
                suggestions.append(f"Retry using the {year} CSV file and explicitly state the year in the answer.")

        if requested_years:
            checks_passed.append(f"Detected requested year(s): {requested_years}.")

        # -----------------------------
        # 4. Cross-year / multi-table validation
        # -----------------------------
        cross_year_requested = (
            ("2014" in q and "2018" in q)
            or "from 2014 to 2018" in q
            or "between 2014 and 2018" in q
            or "cross-year" in q
            or "over time" in q
            or "trend" in q
            or "change" in q
        )

        if cross_year_requested:
            has_2014 = any("2014" in f for f in used_files)
            has_2018 = any("2018" in f for f in used_files)

            if not (has_2014 and has_2018):
                issues.append("The question requires cross-year analysis, but both 2014 and 2018 files were not used.")
                suggestions.append("Load both yearly CSV files and compute changes after aligning/grouping comparable entities.")
            else:
                checks_passed.append("Cross-year analysis used multiple yearly files.")

            if "growth" not in answer_lower and "change" not in answer_lower and "2014" in q and "2018" in q:
                issues.append("The answer does not clearly describe change or growth between 2014 and 2018.")
                suggestions.append("Compute and report absolute change and percentage growth.")

        # -----------------------------
        # 5. Metric validation
        # -----------------------------
        metric_expectations = {
            "revenue": ["revenue"],
            "sales": ["revenue"],
            "net income": ["net income"],
            "earnings": ["net income", "earnings"],
            "profit margin": ["profit margin", "margin"],
            "free cash flow": ["free cash flow", "cash flow"],
            "cash flow": ["cash flow"],
            "roe": ["roe", "return on equity"],
            "roic": ["roic", "return on capital"],
            "debt": ["debt"],
            "current ratio": ["current ratio"],
        }

        for user_term, expected_terms in metric_expectations.items():
            if user_term in q:
                if not any(term in answer_lower for term in expected_terms):
                    issues.append(
                        f"The question asks about {user_term}, but the answer does not clearly analyze that metric."
                    )
                    suggestions.append(f"Retry using the relevant metric for {user_term}.")
                else:
                    checks_passed.append(f"The answer appears to address the requested metric: {user_term}.")
                break

        # -----------------------------
        # 6. Direction validation: highest vs lowest, strongest vs weakest
        # -----------------------------
        asks_low = any(x in q for x in ["lowest", "least", "smallest", "weakest", "worst", "bottom"])
        asks_high = any(x in q for x in ["highest", "most", "largest", "strongest", "best", "top"])

        if asks_low:
            if any(x in answer_lower for x in ["highest", "strongest", "largest", "best"]) and not any(
                x in answer_lower for x in ["lowest", "weakest", "smallest", "bottom"]
            ):
                issues.append("The question asks for the lowest/weakest result, but the answer appears to describe the highest/strongest result.")
                suggestions.append("Retry with ascending sort order or weakest-score ranking.")
            else:
                checks_passed.append("The answer direction appears consistent with a lowest/weakest request.")

        if asks_high:
            if any(x in answer_lower for x in ["lowest", "weakest", "smallest", "worst"]) and not any(
                x in answer_lower for x in ["highest", "strongest", "largest", "top"]
            ):
                issues.append("The question asks for the highest/strongest result, but the answer appears to describe the lowest/weakest result.")
                suggestions.append("Retry with descending sort order or strongest-score ranking.")
            else:
                checks_passed.append("The answer direction appears consistent with a highest/strongest request.")

        # -----------------------------
        # 7. Financial strength validation
        # -----------------------------
        financial_strength_requested = any(
            x in q for x in [
                "financially strongest",
                "financially weakest",
                "financially strong",
                "financially weak",
                "strongest sector",
                "weakest sector",
                "healthiest",
                "riskiest"
            ]
        )

        if financial_strength_requested:
            required_concepts = ["score", "debt", "cash flow", "net income"]
            missing = [c for c in required_concepts if c not in answer_lower]

            if missing:
                issues.append(
                    "Financial strength analysis is incomplete because it does not mention: "
                    + ", ".join(missing)
                )
                suggestions.append(
                    "Use a composite score including profitability, cash flow, liquidity, and leverage."
                )
            else:
                checks_passed.append("Financial strength answer includes multi-metric reasoning.")

        # -----------------------------
        # 8. Visualization validation
        # -----------------------------
        visualization_expected = any(
            x in q for x in [
                "plot",
                "chart",
                "visualize",
                "graph",
                "rank",
                "compare",
                "sector",
                "sectors",
                "relationship",
                "correlation"
            ]
        )

        if visualization_expected and len(plots) == 0:
            issues.append("A visualization is expected for this question, but no Plotly plot was generated.")
            suggestions.append("Generate a Plotly chart and return its JSON.")
        elif len(plots) > 0:
            checks_passed.append("At least one Plotly visualization was generated.")

        # -----------------------------
        # 9. Trace validation
        # -----------------------------
        if len(trace) < 3:
            issues.append("The execution trace has fewer than 3 steps, so the iterative reasoning/execution process is not well demonstrated.")
            suggestions.append("Include inspect, execute, observe, plot, and finalize steps in the trace.")
        else:
            checks_passed.append("The execution trace contains multiple reasoning/execution steps.")

        # -----------------------------
        # Final decision
        # -----------------------------
        if issues:
            status = "retry"
            passed = False
            summary = (
                "Validation failed. The validator detected incomplete or suspicious analysis. "
                "A retry is recommended before accepting this answer."
            )
        else:
            status = "passed"
            passed = True
            summary = (
                "Validation passed. The answer appears consistent with the question, uses appropriate data, "
                "and includes sufficient explanation and visualization support."
            )

        return {
            "status": status,
            "passed": passed,
            "summary": summary,
            "issues": issues,
            "suggestions": suggestions,
            "checks_passed": checks_passed,
            "validation_paragraph": self._build_validation_paragraph(
                passed=passed,
                issues=issues,
                suggestions=suggestions,
                checks_passed=checks_passed
            )
        }

    def _build_validation_paragraph(
        self,
        passed: bool,
        issues: List[str],
        suggestions: List[str],
        checks_passed: List[str]
    ) -> str:
        if passed:
            return (
                "Validation: The validator approved this analysis. It checked that the answer addressed the requested "
                "year, metric, ranking direction, dataset usage, visualization output, and execution trace. "
                "No major inconsistencies were detected."
            )

        issue_text = " ".join(issues[:3])
        suggestion_text = " ".join(suggestions[:2])

        return (
            "Validation: The validator did not fully approve this analysis. "
            f"Main concern(s): {issue_text} "
            f"Recommended correction(s): {suggestion_text}"
        )