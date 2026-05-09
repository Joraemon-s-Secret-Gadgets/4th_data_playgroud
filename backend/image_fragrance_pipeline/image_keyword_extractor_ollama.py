"""
image_keyword_extractor_ollama.py
---------------------------------
Ollama의 qwen2.5vl:7b 같은 오픈소스 VLM으로 이미지에서 시각 키워드를 추출한다.

사전 준비:
    1) Ollama 설치
    2) ollama pull qwen2.5vl:7b
    3) Ollama 서버 실행 상태 확인

사용:
    from image_keyword_extractor_ollama import extract_image_keywords
    result = extract_image_keywords("sample.jpg")
"""

from __future__ import annotations

import base64
import json
import re
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5vl:7b"


IMAGE_ANALYSIS_PROMPT = """
너는 향수 추천 서비스의 이미지 분석기다.

이미지를 보고 향수 추천에 필요한 시각 정보를 추출해라.
향수 이름, 브랜드명, 제품명은 절대 상상해서 만들지 마라.
이미지에 보이는 색감, 사물, 장면, 분위기만 분석해라.

반드시 JSON 객체 하나만 출력해라.
마크다운, 설명 문장, 코드블록은 절대 출력하지 마라.
같은 키워드를 반복하지 마라.

출력 형식:
{
  "visual_summary": "이미지 전체 분위기를 한국어 한 문장으로 요약",
  "colors": [],
  "objects": [],
  "scene": [],
  "mood": [],
  "season": [],
  "time": [],
  "raw_keywords": []
}

필드별 규칙:
- visual_summary: 한국어 한 문장
- colors: black, brown, white, green, blue, pink, red, gold, beige, gray 중 최대 5개
- objects: flower, wood, leather, ocean, book, coffee, candle, fabric, glass, tree, fruit, metal, stone, dress, handbag, necklace 중 최대 6개
- scene: forest, beach, cafe, bar, city, room, garden, rainy street, office, bedroom, outdoor, indoor 중 최대 3개
- mood: clean, warm, dark, soft, luxurious, natural, urban, calm, romantic, sensual, fresh, bright, cozy, modern 중 최대 4개
- season: spring, summer, autumn, winter 중 최대 1개
- time: morning, afternoon, evening, night 중 최대 1개
- raw_keywords: 이미지 보조 키워드 최대 8개. 절대 반복하지 마라.

중요:
- 배열은 반드시 JSON 배열로 작성해라.
- scene, mood, season, time도 문자열이 아니라 배열로 작성해라.
- raw_keywords를 길게 쓰지 마라.
- raw_keywords 안에 비슷한 표현을 반복하지 마라.
- JSON의 마지막 }까지 반드시 닫아라.
"""


REQUIRED_KEYS = [
    "visual_summary",
    "colors",
    "objects",
    "scene",
    "mood",
    "season",
    "time",
    "raw_keywords",
]


MAX_ITEMS = {
    "colors": 5,
    "objects": 6,
    "scene": 3,
    "mood": 4,
    "season": 1,
    "time": 1,
    "raw_keywords": 8,
}


def encode_image_to_base64(
    image_path: str | Path,
    max_size: int = 768,
    quality: int = 85,
    show_progress: bool = True,
) -> str:
    """
    이미지를 Ollama에 보내기 좋게 자동 리사이즈 후 base64로 변환한다.
    """

    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {path}")

    img = Image.open(path).convert("RGB")
    original_size = img.size

    img.thumbnail((max_size, max_size))

    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=quality)

    resized_size = img.size

    if show_progress:
        print(f"[이미지 리사이즈] {original_size} → {resized_size}")
        print("[이미지 인코딩] base64 변환 완료")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _dedupe_list(values: list[str], max_items: int | None = None) -> list[str]:
    seen = set()
    result = []

    for value in values:
        cleaned = str(value).strip().lower()
        if not cleaned:
            continue

        if cleaned in seen:
            continue

        seen.add(cleaned)
        result.append(cleaned)

        if max_items is not None and len(result) >= max_items:
            break

    return result


def _ensure_list(value: Any, key: str) -> list[str]:
    """
    모델 출력이 문자열, None, 리스트 중 어떤 형태여도 안정적으로 list[str]로 변환한다.
    """

    if value is None:
        values = []
    elif isinstance(value, list):
        values = [str(v).strip().lower() for v in value if str(v).strip()]
    elif isinstance(value, str):
        values = [value.strip().lower()] if value.strip() else []
    else:
        values = [str(value).strip().lower()] if str(value).strip() else []

    return _dedupe_list(values, MAX_ITEMS.get(key))


def _normalize_result(data: dict[str, Any]) -> dict[str, Any]:
    """
    이미지 분석 결과의 키와 타입을 고정한다.
    """

    normalized = {
        "visual_summary": str(data.get("visual_summary", "")).strip(),
        "colors": _ensure_list(data.get("colors", []), "colors"),
        "objects": _ensure_list(data.get("objects", []), "objects"),
        "scene": _ensure_list(data.get("scene", []), "scene"),
        "mood": _ensure_list(data.get("mood", []), "mood"),
        "season": _ensure_list(data.get("season", []), "season"),
        "time": _ensure_list(data.get("time", []), "time"),
        "raw_keywords": _ensure_list(data.get("raw_keywords", []), "raw_keywords"),
    }

    return normalized


def _extract_json_object_text(text: str) -> str:
    """
    모델이 JSON 앞뒤에 텍스트를 붙였을 때 가장 바깥 JSON 객체 텍스트만 추출한다.
    완전한 JSON일 때만 성공한다.
    """

    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()

    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()

    if text.endswith("```"):
        text = text[: -3].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or start >= end:
        raise ValueError("완전한 JSON 객체를 찾을 수 없습니다.")

    return text[start : end + 1]


def _parse_json_or_raise(text: str) -> dict[str, Any]:
    """
    정상 JSON 파싱을 먼저 시도하고, 실패하면 JSON 객체 텍스트 추출 후 다시 시도한다.
    """

    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    json_text = _extract_json_object_text(text)
    return json.loads(json_text)


def _extract_string_field(text: str, key: str) -> str:
    """
    깨진 JSON에서 문자열 필드를 최대한 복구한다.
    예: "visual_summary": "..."
    """

    pattern = rf'"{re.escape(key)}"\s*:\s*"([^"]*)"' 
    match = re.search(pattern, text, re.DOTALL)

    if not match:
        return ""

    return match.group(1).strip()


def _extract_array_field(text: str, key: str, max_items: int) -> list[str]:
    """
    깨진 JSON에서 배열 필드를 최대한 복구한다.

    정상 예:
      "colors": ["green", "black"]

    중간에 잘린 예:
      "raw_keywords": ["plaid", "romantic", "romantic dress"

    이 경우에도 따옴표 안의 값들을 최대한 추출한다.
    """

    key_pattern = rf'"{re.escape(key)}"\s*:\s*'
    key_match = re.search(key_pattern, text)

    if not key_match:
        # 모델이 문자열로 준 경우도 처리
        string_value = _extract_string_field(text, key)
        return _dedupe_list([string_value], max_items) if string_value else []

    start = key_match.end()
    remaining = text[start:].lstrip()

    # 배열이 아닌 문자열로 나온 경우
    if remaining.startswith('"'):
        value_match = re.match(r'"([^"]*)"', remaining, re.DOTALL)
        if value_match:
            return _dedupe_list([value_match.group(1)], max_items)
        return []

    # 배열이 아닌 경우
    if not remaining.startswith("["):
        return []

    # 다음 필드가 시작되기 전까지 잘라서 분석
    next_field_match = re.search(
        r',\s*"(colors|objects|scene|mood|season|time|raw_keywords|visual_summary)"\s*:',
        remaining[1:],
    )

    if next_field_match:
        chunk = remaining[: next_field_match.start() + 1]
    else:
        chunk = remaining

    # 따옴표 안의 값만 추출
    values = re.findall(r'"([^"]+)"', chunk)

    return _dedupe_list(values, max_items)


def _fallback_parse_partial_json(text: str) -> dict[str, Any]:
    """
    모델 출력이 반복/중단되어 JSON이 깨졌을 때도 최대한 필요한 필드를 복구한다.
    """

    recovered = {
        "visual_summary": _extract_string_field(text, "visual_summary"),
        "colors": _extract_array_field(text, "colors", MAX_ITEMS["colors"]),
        "objects": _extract_array_field(text, "objects", MAX_ITEMS["objects"]),
        "scene": _extract_array_field(text, "scene", MAX_ITEMS["scene"]),
        "mood": _extract_array_field(text, "mood", MAX_ITEMS["mood"]),
        "season": _extract_array_field(text, "season", MAX_ITEMS["season"]),
        "time": _extract_array_field(text, "time", MAX_ITEMS["time"]),
        "raw_keywords": _extract_array_field(text, "raw_keywords", MAX_ITEMS["raw_keywords"]),
    }

    return _normalize_result(recovered)


def _parse_ollama_stream_response(
    response: requests.Response,
    show_progress: bool = True,
) -> str:
    """
    Ollama stream 응답을 실시간으로 읽어서 최종 텍스트를 합친다.
    """

    chunks: list[str] = []

    for line in response.iter_lines():
        if not line:
            continue

        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        token = data.get("response", "")

        if token:
            chunks.append(token)

            if show_progress:
                print(token, end="", flush=True)

        if data.get("done"):
            if show_progress:
                print("\n[완료] Ollama 응답 생성 완료")
            break

    return "".join(chunks).strip()


def extract_image_keywords(
    image_path: str | Path,
    model: str = DEFAULT_MODEL,
    ollama_url: str = OLLAMA_URL,
    timeout: int = 600,
    max_image_size: int = 768,
    image_quality: int = 85,
    show_progress: bool = True,
) -> dict[str, Any]:
    """
    이미지 한 장을 분석해 향수 추천에 필요한 시각 키워드를 추출한다.
    """

    if show_progress:
        print("[1/4] 이미지 준비 시작")

    image_base64 = encode_image_to_base64(
        image_path=image_path,
        max_size=max_image_size,
        quality=image_quality,
        show_progress=show_progress,
    )

    payload = {
        "model": model,
        "prompt": IMAGE_ANALYSIS_PROMPT,
        "images": [image_base64],
        "stream": True,
        "format": "json",
        "options": {
            "temperature": 0.0,
            "top_p": 0.8,
            "repeat_penalty": 1.25,
            "num_predict": 384,
        },
    }

    if show_progress:
        print("[2/4] Ollama 요청 준비 완료")
        print(f"[모델] {model}")
        print(f"[엔드포인트] {ollama_url}")
        print("[3/4] 모델 응답 생성 중...\n")

    try:
        with requests.post(
            ollama_url,
            json=payload,
            timeout=timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            text = _parse_ollama_stream_response(
                response=response,
                show_progress=show_progress,
            )

    except requests.exceptions.ConnectionError as e:
        raise ConnectionError(
            "Ollama 서버에 연결하지 못했습니다. "
            "먼저 `ollama serve`를 실행하거나 Ollama 앱이 켜져 있는지 확인하세요."
        ) from e

    except requests.exceptions.ReadTimeout as e:
        raise TimeoutError(
            f"Ollama 응답이 {timeout}초 안에 끝나지 않았습니다. "
            "이미지 크기를 더 줄이거나, 더 가벼운 Vision 모델을 사용해보세요."
        ) from e

    if show_progress:
        print("[4/4] JSON 파싱 시작")

    try:
        parsed = _parse_json_or_raise(text)
        result = _normalize_result(parsed)

    except Exception:
        if show_progress:
            print("[경고] JSON이 깨져서 fallback 복구 파서를 사용합니다.")

        result = _fallback_parse_partial_json(text)

    if show_progress:
        print("[완료] 이미지 키워드 추출 완료")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ollama Vision 모델로 이미지 키워드를 추출합니다."
    )
    parser.add_argument("image_path", help="분석할 이미지 파일 경로")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Ollama 모델명. 기본값: {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--out",
        default="",
        help="결과를 저장할 JSON 파일 경로",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Ollama 응답 대기 시간. 기본값: 600초",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=768,
        help="이미지 긴 변 최대 크기. 기본값: 768",
    )

    args = parser.parse_args()

    result = extract_image_keywords(
        image_path=args.image_path,
        model=args.model,
        timeout=args.timeout,
        max_image_size=args.max_size,
        show_progress=True,
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[저장 완료] {args.out}")