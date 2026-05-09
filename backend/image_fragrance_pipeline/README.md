# Image Fragrance Keyword Pipeline

이미지 한 장을 입력받아 향수 추천/RAG 검색용 키워드로 변환하는 코드입니다.

## 1. 설치

```bash
pip install -r requirements_image.txt
```

Ollama 설치 후 모델을 받습니다.

```bash
ollama pull qwen2.5vl:7b
```

Ollama 서버가 켜져 있어야 합니다.

```bash
ollama serve
```

이미 실행 중이면 생략해도 됩니다.

## 2. 실행

```bash
python test_image_pipeline.py ./sample.jpg
```

결과 저장:

```bash
python test_image_pipeline.py ./sample.jpg --out image_keyword_result.json
```

## 3. 출력 구조

```json
{
  "image_keywords": {
    "visual_summary": "...",
    "colors": [],
    "objects": [],
    "scene": [],
    "mood": [],
    "season": [],
    "time": [],
    "raw_keywords": []
  },
  "fragrance_keywords": {
    "matched_triggers": [],
    "fragrance_families": [],
    "fragrance_subs": [],
    "components": [],
    "components_ko": [],
    "descriptors": [],
    "query_text": "..."
  },
  "query_text": "..."
}
```

`query_text`를 나중에 RAG 검색 질의로 사용하면 됩니다.

## 4. 파일 설명

- `image_keyword_extractor_ollama.py`: Ollama VLM으로 이미지의 색감/객체/장면/분위기를 추출
- `image_to_fragrance_mapper.py`: 시각 키워드를 향수 family/sub/component/descriptor로 변환
- `fragrance_aliases.py`: 레더/가죽향/leather 같은 표기를 canonical 키워드로 통합
- `image_pipeline.py`: 전체 파이프라인 함수
- `test_image_pipeline.py`: CLI 테스트
