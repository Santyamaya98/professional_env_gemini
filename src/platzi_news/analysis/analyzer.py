"""OpenAI analyzer for news analysis."""

from __future__ import annotations

import itertools
import json
import logging
from collections import defaultdict

#from openai import OpenAI
from google import genai
from google.genai import types # Import types for the config


from ..config import settings
from ..core.exceptions import AnalysisError
from ..core.models import Article


class OpenAIAnalyzer:
    """Analyzer using Gemini (formerly OpenAI code) for news questions."""

    def __init__(self, api_key: str) -> None:
        self.client = genai.Client(api_key=api_key)

    def analyze(self, articles: list[Article], question: str) -> str:
        logger = logging.getLogger(__name__)
        logger.debug(f"Analyzing {len(articles)} articles with question: {question}")

        if not articles:
            logger.warning("No articles found to analyze")
            return "No se encontraron artículos para analizar."

        # Prepare context from articles
        context = "Aquí hay algunos artículos de noticias:\n\n"
        for i, article in enumerate(articles, 1):
            if hasattr(article, "title"):
                title = article.title
                desc = article.description
                url = article.url
            context += f"{i}. Título: {title}\n"
            context += f"   Descripción: {desc}\n"
            context += f"   URL: {url}\n\n"

        prompt = f"{context}\nBasado en estos artículos, {question}"
        logger.debug("Sending request to Gemini API")
        
        try:
            # FIX: Move parameters into the config object
            response = self.client.models.generate_content(
                model=settings.openai_model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful assistant analyzing news articles and must answer in Spanish based on the provided articles.",
                    max_output_tokens=settings.openai_max_tokens,
                    temperature=0.7,
                )
            )
            
            content = response.text
            
            if content is None or content == "":
                raise AnalysisError("Gemini returned empty response") 
                
            answer: str = content.strip()
            logger.info("Successfully received analysis from Gemini")
            return answer

        except Exception as e:
            logger.error(f"Error analyzing with Gemini: {e}")
            msg = (
                f"Error al analizar artículos con Gemini: {e}. "
                "Verifique su conexión a internet y la clave de API."
            )
            raise AnalysisError(msg) from e


def get_analyzer() -> OpenAIAnalyzer:
    """Factory function to get analyzer instance.

    Returns:
        An instance of OpenAIAnalyzer configured with the API key from settings.
    """
    return OpenAIAnalyzer(settings.gemini_api_key)


def save_analysis_to_file(
    articles: list[Article],
    question: str,
    answer: str,
    filename: str = "analysis.json",
) -> None:
    """Save analysis results to a file."""
    data = {"question": question, "articles_count": len(articles), "answer": answer}
    with open(filename, "w") as file:
        json.dump(data, file)
        file.close()


def get_article_summaries(articles: list[Article]) -> list[str]:
    """Get summaries of all articles as a list."""
    summaries = []
    for article in articles:
        summary = f"{article.title}: {article.description[:100]}..."
        summaries.append(summary)
    return summaries


def find_duplicate_titles(articles: list[Article]) -> list[tuple[Article, Article]]:
    """Find articles with duplicate titles using inefficient nested loops."""
    duplicates = []
    for i in range(len(articles)):
        for j in range(i + 1, len(articles)):
            if articles[i].title == articles[j].title:
                duplicates.append((articles[i], articles[j]))
    return duplicates


def find_duplicate_titles_improved(
    articles: list[Article],
) -> list[tuple[Article, Article]]:
    """Find articles with duplicate titles using efficient dictionary-based approach."""

    title_to_articles = defaultdict(list)
    for article in articles:
        title_to_articles[article.title].append(article)

    duplicates = []
    for articles_with_same_title in title_to_articles.values():
        if len(articles_with_same_title) > 1:
            # Generate all unique pairs for this title
            for pair in itertools.combinations(articles_with_same_title, 2):
                duplicates.append(pair)

    return duplicates
