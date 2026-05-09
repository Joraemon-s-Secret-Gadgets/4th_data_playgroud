"""
image_pipeline.py
-----------------
이미지 한 장을 입력받아:
1) 시각 키워드 추출
2) 향수 도메인 키워드 변환
3) RAG 검색용 query_text 생성
까지 수행한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from image_keyword_extractor_ollama import extract_image_keywords
from image_to_fragrance_mapper import map_image_to_fragrance_keywords


def analyze_image_for_fragrance(
    image_path: str | Path,
    model: str = "qwen2.5vl:7b",
) -> dict[str, Any]:
    image_keywords = extract_image_keywords(image_path=image_path, model=model)
    fragrance_keywords = map_image_to_fragrance_keywords(image_keywords)

    return {
        "image_path": str(image_path),
        "image_keywords": image_keywords,
        "fragrance_keywords": fragrance_keywords,
        "query_text": fragrance_keywords.get("query_text", ""),
    }


def save_result(result: dict[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
