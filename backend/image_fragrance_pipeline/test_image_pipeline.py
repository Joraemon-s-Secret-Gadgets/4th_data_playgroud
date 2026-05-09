"""
CLI 테스트용.

실행:
    python test_image_pipeline.py ./sample.jpg
    python test_image_pipeline.py ./sample.jpg --model qwen2.5vl:7b --out result.json
"""

from __future__ import annotations

import argparse
import json

from image_pipeline import analyze_image_for_fragrance, save_result


def main() -> None:
    parser = argparse.ArgumentParser(description="이미지에서 향수 추천용 키워드 추출")
    parser.add_argument("image_path", help="분석할 이미지 경로")
    parser.add_argument("--model", default="qwen2.5vl:7b", help="Ollama VLM 모델명")
    parser.add_argument("--out", default="", help="결과 JSON 저장 경로")
    args = parser.parse_args()

    result = analyze_image_for_fragrance(args.image_path, model=args.model)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        save_result(result, args.out)
        print(f"\n저장 완료: {args.out}")


if __name__ == "__main__":
    main()
