from typing import List, Dict, Any
import os

try:
    from duckduckgo_search import DDGS
except Exception:
    DDGS = None

try:
    from dotenv import load_dotenv
    from openai import OpenAI
except Exception:
    load_dotenv = None
    OpenAI = None


class WebSearchAgent:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

        if load_dotenv:
            load_dotenv()

        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.client = None

        if OpenAI and os.getenv("OPENAI_API_KEY"):
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _llm_rewrite_queries(self, question: str) -> List[str]:
        if not self.client:
            return [question]

        prompt = f"""
Create 3 effective web search queries for answering this question.
Avoid overly literal wording. Include key entities and concepts.

Question:
{question}

Return only a JSON list of strings.
"""

        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0
            )

            text = res.choices[0].message.content.strip()
            import json
            queries = json.loads(text)

            if isinstance(queries, list) and queries:
                return [str(q) for q in queries]

        except Exception as e:
            print("LLM query rewrite failed:", e)

        return [question]

    def search(self, question: str) -> List[Dict[str, str]]:
        if DDGS is None:
            return []

        queries = self._llm_rewrite_queries(question)
        results = []

        bad_domains = [
            "dictionary",
            "thesaurus",
            "wordreference",
            "merriam-webster",
            "freedictionary"
        ]

        for query in queries:
            try:
                with DDGS() as ddgs:
                    raw = list(ddgs.text(query, max_results=10))

                for item in raw:
                    result = {
                        "title": item.get("title", ""),
                        "href": item.get("href", ""),
                        "body": item.get("body", ""),
                        "query_used": query
                    }

                    text = f"{result['title']} {result['href']} {result['body']}".lower()

                    if any(bad in text for bad in bad_domains):
                        continue

                    results.append(result)

                if len(results) >= self.max_results:
                    break

            except Exception as e:
                print("DuckDuckGo search failed:", e)

        return results[:self.max_results]

    def _llm_answer(self, question: str, results: List[Dict[str, str]]) -> str:
        if not self.client:
            return ""

        sources_text = "\n\n".join(
            f"Source {i}\nTitle: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}"
            for i, r in enumerate(results, start=1)
        )

        prompt = f"""
Answer the user's question directly and clearly.

Use the search results if they are relevant.
If search results are weak or missing, give a careful general answer and say that search evidence was limited.

Question:
{question}

Search results:
{sources_text}

Requirements:
- Start with the direct answer.
- Then explain the reasoning in 1-3 short paragraphs.
- If sources are available, include a short Sources section with URLs.
"""

        try:
            res = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a concise web-grounded research assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )

            return res.choices[0].message.content.strip()

        except Exception as e:
            print("LLM answer synthesis failed:", e)
            return ""

    def _fallback_answer(self, question: str, results: List[Dict[str, str]]) -> str:
        if not results:
            return (
                "I could not retrieve useful web results for this question. "
                "The system needs either internet access or an LLM answer-synthesis step to answer this reliably."
            )

        answer = f"Here is a search-grounded answer for: {question}\n\n"

        for i, r in enumerate(results, start=1):
            answer += f"{i}. {r['title']}\n{r['body']}\nURL: {r['href']}\n\n"

        return answer

    def answer(self, question: str) -> Dict[str, Any]:
        queries = self._llm_rewrite_queries(question)
        results = self.search(question)

        final_answer = self._llm_answer(question, results)

        if not final_answer:
            final_answer = self._fallback_answer(question, results)

        return {
            "route": "web",
            "final_answer": final_answer,
            "sources": results,
            "plots": [],
            "trace": [
                {
                    "step": "query_rewrite",
                    "original_question": question,
                    "query_variants_used": queries
                },
                {
                    "step": "duckduckgo_search",
                    "num_results": len(results),
                    "first_result": results[0]["title"] if results else None
                },
                {
                    "step": "answer_synthesis",
                    "observation": "Generated a direct answer using LLM synthesis when available."
                }
            ]
        }