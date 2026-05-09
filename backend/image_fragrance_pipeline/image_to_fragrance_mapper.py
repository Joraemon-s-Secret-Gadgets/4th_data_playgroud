"""
image_to_fragrance_mapper.py
----------------------------
이미지 시각 키워드를 향수 도메인 키워드로 변환한다.

역할:
1. 이미지 키워드 colors / objects / scene / mood / raw_keywords를 수집
2. 시각 키워드를 향수 family / sub / component / descriptor로 점수화
3. 패션/봄/오후/여성적 이미지에서는 Fresh/Floral 쪽을 보정
4. black 단독으로 Leather/Oud/Incense가 과하게 올라가지 않도록 보정
5. 최종 query_text를 생성해 RAG 검색 질의로 사용

중요:
- query_text는 RAG 검색용이며 prefix를 붙인다.
  예: visual_color:green / fragrance_sub:Green
- readable_query_text는 사람이 보기 좋은 디버깅용이다.
- query_sections는 나중에 가중치 조정/로그 분석용으로 사용한다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from fragrance_aliases import normalize_note_keyword, to_korean_note


def _mapping(
    families: dict[str, float],
    subs: dict[str, float],
    components: dict[str, float],
    descriptors: dict[str, float],
) -> dict[str, dict[str, float]]:
    return {
        "families": families,
        "subs": subs,
        "components": components,
        "descriptors": descriptors,
    }


VISUAL_TO_FRAGRANCE_RULES: dict[str, dict[str, dict[str, float]]] = {
    # -----------------------------------------------------
    # colors
    # -----------------------------------------------------
    "black": _mapping(
        families={"WOODY": 0.8, "AMBERY": 0.5},
        subs={"Dry Woods": 0.6, "Soft Amber": 0.5},
        components={
            "Leather": 0.5,
            "Incense": 0.4,
            "Oud": 0.4,
            "Patchouli": 0.3,
            "Musk": 0.5,
        },
        descriptors={
            "깊은": 0.7,
            "차분한": 0.6,
            "세련된": 0.6,
            "묵직한": 0.3,
            "스모키한": 0.3,
        },
    ),
    "brown": _mapping(
        families={"WOODY": 1.4, "AMBERY": 1.0},
        subs={"Woods": 1.2, "Woody Amber": 1.0, "Dry Woods": 0.6},
        components={
            "Sandalwood": 1.3,
            "Cedarwood": 1.1,
            "Amber": 1.0,
            "Tobacco": 0.6,
        },
        descriptors={
            "따뜻한": 1.3,
            "안정감": 1.0,
            "깊은": 0.9,
            "묵직한": 0.5,
        },
    ),
    "gold": _mapping(
        families={"AMBERY": 1.6, "FLORAL": 0.5},
        subs={"Amber": 1.4, "Soft Amber": 1.1, "Floral Amber": 0.7},
        components={
            "Amber": 1.6,
            "Vanilla": 1.0,
            "Musk": 0.8,
        },
        descriptors={
            "따뜻한": 1.2,
            "포근한": 0.9,
            "찬란한": 1.1,
            "매력적인": 1.0,
        },
    ),
    "green": _mapping(
        families={"FRESH": 1.8, "WOODY": 0.7},
        subs={"Green": 1.8, "Mossy Woods": 0.8, "Aromatic": 0.7},
        components={
            "Green Note": 1.8,
            "Oakmoss": 0.9,
            "Vetiver": 0.8,
            "Basil": 0.8,
            "Bergamot": 0.5,
        },
        descriptors={
            "청량한": 1.5,
            "신선한": 1.5,
            "상쾌한": 1.3,
            "청정한": 1.0,
        },
    ),
    "blue": _mapping(
        families={"FRESH": 1.8},
        subs={"Water": 1.5, "Citrus": 0.8},
        components={
            "Marine": 1.4,
            "Ozonic": 1.2,
            "Bergamot": 0.8,
            "Lemon": 0.7,
            "Musk": 0.4,
        },
        descriptors={
            "시원한": 1.5,
            "상쾌한": 1.3,
            "가벼운": 1.0,
            "깨끗한": 0.9,
        },
    ),
    "white": _mapping(
        families={"FRESH": 1.3, "FLORAL": 1.0, "AMBERY": 0.4},
        subs={"Green": 0.8, "Soft Floral": 1.1, "Soft Amber": 0.6},
        components={
            "Musk": 1.3,
            "Iris": 1.0,
            "Lily": 0.9,
            "Bergamot": 0.5,
        },
        descriptors={
            "깨끗한": 1.5,
            "부드러운": 1.3,
            "맑은": 1.0,
            "가벼운": 0.9,
        },
    ),
    "pink": _mapping(
        families={"FLORAL": 1.7, "FRESH": 0.8},
        subs={"Floral": 1.5, "Soft Floral": 1.0, "Fruity": 0.8},
        components={
            "Absolute Rose": 1.5,
            "Peony": 1.3,
            "Berry, Apple, Peach": 0.8,
            "Musk": 0.4,
        },
        descriptors={
            "우아한": 1.2,
            "달콤한": 1.0,
            "밝은": 1.0,
            "감미로운": 1.0,
            "부드러운": 0.8,
        },
    ),
    "red": _mapping(
        families={"FLORAL": 1.0, "AMBERY": 1.0, "FRESH": 0.5},
        subs={"Floral Amber": 1.3, "Fruity": 1.0, "Amber": 0.7},
        components={
            "Absolute Rose": 1.2,
            "Pepper": 0.9,
            "Spicy": 0.9,
            "Berry, Apple, Peach": 1.0,
        },
        descriptors={
            "감각적인": 1.3,
            "달콤한": 1.1,
            "강한": 1.0,
            "매력적인": 1.0,
        },
    ),
    "beige": _mapping(
        families={"FLORAL": 0.9, "AMBERY": 1.1, "WOODY": 0.5},
        subs={"Soft Floral": 1.0, "Soft Amber": 1.2, "Woods": 0.4},
        components={
            "Musk": 1.2,
            "Iris": 1.0,
            "Vanilla": 1.0,
            "Sandalwood": 0.7,
        },
        descriptors={
            "부드러운": 1.5,
            "포근한": 1.2,
            "온화한": 1.0,
            "따뜻한": 0.8,
        },
    ),
    "gray": _mapping(
        families={"FRESH": 0.8, "WOODY": 0.6, "AMBERY": 0.4},
        subs={"Water": 0.8, "Mossy Woods": 0.6, "Soft Amber": 0.4},
        components={
            "Musk": 1.0,
            "Vetiver": 0.7,
            "Ozonic": 0.8,
            "Oakmoss": 0.5,
        },
        descriptors={
            "차분한": 1.2,
            "깨끗한": 1.0,
            "시원한": 0.8,
        },
    ),

    # -----------------------------------------------------
    # objects
    # -----------------------------------------------------
    "dress": _mapping(
        families={"FLORAL": 1.2, "FRESH": 1.0},
        subs={"Soft Floral": 1.1, "Floral": 1.0, "Green": 0.7},
        components={
            "Peony": 1.1,
            "Absolute Rose": 1.0,
            "Musk": 0.9,
            "Iris": 0.7,
            "Bergamot": 0.6,
        },
        descriptors={
            "우아한": 1.2,
            "부드러운": 1.1,
            "밝은": 0.8,
            "감미로운": 0.8,
        },
    ),
    "handbag": _mapping(
        families={"FLORAL": 0.8, "AMBERY": 0.6, "FRESH": 0.6},
        subs={"Soft Floral": 0.8, "Soft Amber": 0.6},
        components={
            "Musk": 0.9,
            "Iris": 0.7,
            "Absolute Rose": 0.6,
        },
        descriptors={
            "세련된": 1.0,
            "우아한": 0.9,
            "부드러운": 0.6,
        },
    ),
    "necklace": _mapping(
        families={"FLORAL": 0.8, "AMBERY": 0.6},
        subs={"Soft Floral": 0.7, "Floral Amber": 0.5},
        components={
            "Musk": 0.7,
            "Absolute Rose": 0.6,
            "Amber": 0.5,
        },
        descriptors={
            "우아한": 1.0,
            "찬란한": 0.6,
            "매력적인": 0.7,
        },
    ),
    "wood": _mapping(
        families={"WOODY": 2.0},
        subs={"Woods": 1.6, "Dry Woods": 0.9, "Woody Amber": 0.8},
        components={
            "Wood Scent": 1.5,
            "Cedarwood": 1.4,
            "Sandalwood": 1.4,
            "Vetiver": 1.0,
        },
        descriptors={
            "따뜻한": 1.1,
            "안정감": 1.1,
            "깊은": 1.0,
            "묵직한": 0.7,
        },
    ),
    "tree": _mapping(
        families={"WOODY": 1.3, "FRESH": 1.0},
        subs={"Woods": 1.0, "Green": 1.0},
        components={
            "Pine tree": 1.0,
            "Cedarwood": 0.9,
            "Green Note": 1.0,
            "Vetiver": 0.8,
        },
        descriptors={
            "신선한": 1.0,
            "상쾌한": 0.9,
            "안정감": 0.8,
            "깊은": 0.6,
        },
    ),
    "leather": _mapping(
        families={"WOODY": 2.0, "AMBERY": 0.8},
        subs={"Dry Woods": 2.0, "Woody Amber": 0.8},
        components={
            "Leather": 2.5,
            "Tobacco": 1.0,
            "Incense": 1.0,
            "Saffron": 0.8,
        },
        descriptors={
            "스모키한": 1.7,
            "따뜻한": 1.2,
            "묵직한": 1.5,
            "감각적인": 1.3,
        },
    ),
    "flower": _mapping(
        families={"FLORAL": 2.0},
        subs={"Floral": 1.6, "Soft Floral": 1.0},
        components={
            "Absolute Rose": 1.3,
            "Absolute Jasmin": 1.3,
            "Peony": 1.1,
            "Lily": 0.9,
        },
        descriptors={
            "우아한": 1.3,
            "섬세한": 1.2,
            "부드러운": 1.1,
            "감미로운": 1.0,
        },
    ),
    "ocean": _mapping(
        families={"FRESH": 2.0},
        subs={"Water": 1.8, "Citrus": 0.6},
        components={
            "Marine": 1.8,
            "Ozonic": 1.4,
            "Bergamot": 0.6,
        },
        descriptors={
            "시원한": 1.6,
            "짭짤한": 1.2,
            "축축한": 1.0,
            "청량한": 1.2,
        },
    ),
    "coffee": _mapping(
        families={"AMBERY": 1.7},
        subs={"Soft Amber": 1.2, "Amber": 1.0},
        components={
            "Vanilla": 1.2,
            "Amber": 1.1,
            "Musk": 0.8,
        },
        descriptors={
            "따뜻한": 1.4,
            "포근한": 1.2,
            "달콤한": 1.2,
            "깊은": 1.0,
        },
    ),
    "candle": _mapping(
        families={"AMBERY": 1.4, "WOODY": 0.8},
        subs={"Soft Amber": 1.2, "Dry Woods": 0.8},
        components={
            "Incense": 1.4,
            "Amber": 1.2,
            "Musk": 0.8,
        },
        descriptors={
            "신비로운": 1.2,
            "따뜻한": 1.1,
            "포근한": 1.1,
            "차분한": 1.0,
        },
    ),
    "book": _mapping(
        families={"WOODY": 1.0, "AMBERY": 0.6},
        subs={"Woods": 0.9, "Soft Amber": 0.7},
        components={
            "Cedarwood": 0.8,
            "Sandalwood": 0.8,
            "Musk": 0.6,
        },
        descriptors={
            "차분한": 1.2,
            "부드러운": 0.9,
            "깊은": 0.8,
        },
    ),
    "fabric": _mapping(
        families={"FRESH": 0.8, "FLORAL": 0.8, "AMBERY": 0.5},
        subs={"Soft Floral": 0.9, "Soft Amber": 0.7},
        components={
            "Musk": 1.0,
            "Iris": 0.8,
            "Lily": 0.6,
        },
        descriptors={
            "부드러운": 1.2,
            "깨끗한": 1.0,
            "포근한": 0.8,
        },
    ),
    "fruit": _mapping(
        families={"FRESH": 1.6},
        subs={"Fruity": 1.6, "Citrus": 0.8},
        components={
            "Berry, Apple, Peach": 1.4,
            "Tropical Fruit": 1.0,
            "Orange": 0.8,
        },
        descriptors={
            "달콤한": 1.4,
            "밝은": 1.1,
            "시원한": 0.8,
        },
    ),

    # -----------------------------------------------------
    # scenes
    # -----------------------------------------------------
    "indoor": _mapping(
        families={"AMBERY": 0.3, "FLORAL": 0.3},
        subs={"Soft Amber": 0.3, "Soft Floral": 0.3},
        components={
            "Musk": 0.4,
            "Iris": 0.3,
        },
        descriptors={
            "차분한": 0.4,
            "부드러운": 0.3,
        },
    ),
    "room": _mapping(
        families={"AMBERY": 0.3, "FLORAL": 0.3},
        subs={"Soft Amber": 0.3, "Soft Floral": 0.3},
        components={
            "Musk": 0.4,
            "Iris": 0.3,
        },
        descriptors={
            "차분한": 0.4,
            "부드러운": 0.3,
        },
    ),
    "forest": _mapping(
        families={"WOODY": 1.8, "FRESH": 1.2},
        subs={"Mossy Woods": 1.6, "Green": 1.2, "Woods": 0.8},
        components={
            "Oakmoss": 1.5,
            "Vetiver": 1.3,
            "Green Note": 1.2,
            "Pine tree": 1.0,
        },
        descriptors={
            "신선한": 1.2,
            "상쾌한": 1.1,
            "깊은": 1.0,
            "묵직한": 0.8,
        },
    ),
    "beach": _mapping(
        families={"FRESH": 2.0},
        subs={"Water": 1.6, "Citrus": 1.0},
        components={
            "Marine": 1.6,
            "Ozonic": 1.2,
            "Bergamot": 0.9,
            "Lemon": 0.8,
        },
        descriptors={
            "시원한": 1.5,
            "청량한": 1.3,
            "상쾌한": 1.2,
            "가벼운": 1.0,
        },
    ),
    "cafe": _mapping(
        families={"AMBERY": 1.5, "WOODY": 0.5},
        subs={"Soft Amber": 1.3, "Amber": 0.8},
        components={
            "Vanilla": 1.4,
            "Amber": 1.0,
            "Musk": 0.8,
            "Sandalwood": 0.6,
        },
        descriptors={
            "따뜻한": 1.4,
            "포근한": 1.3,
            "달콤한": 1.1,
            "부드러운": 1.0,
        },
    ),
    "bar": _mapping(
        families={"WOODY": 1.5, "AMBERY": 1.5},
        subs={"Dry Woods": 1.4, "Amber": 1.2, "Woody Amber": 1.0},
        components={
            "Leather": 1.5,
            "Tobacco": 1.4,
            "Amber": 1.2,
            "Incense": 1.0,
        },
        descriptors={
            "스모키한": 1.3,
            "묵직한": 1.3,
            "감각적인": 1.2,
            "깊은": 1.0,
        },
    ),
    "garden": _mapping(
        families={"FLORAL": 1.5, "FRESH": 1.0},
        subs={"Floral": 1.3, "Green": 0.9, "Soft Floral": 0.8},
        components={
            "Absolute Rose": 1.2,
            "Peony": 1.0,
            "Green Note": 0.9,
            "Bergamot": 0.7,
        },
        descriptors={
            "우아한": 1.1,
            "신선한": 1.0,
            "상쾌한": 0.9,
            "밝은": 0.8,
        },
    ),
    "rainy street": _mapping(
        families={"FRESH": 1.4, "WOODY": 0.9},
        subs={"Water": 1.3, "Mossy Woods": 1.0},
        components={
            "Ozonic": 1.4,
            "Marine": 0.9,
            "Oakmoss": 1.0,
            "Vetiver": 0.8,
        },
        descriptors={
            "축축한": 1.4,
            "시원한": 1.2,
            "깊은": 0.8,
            "청정한": 0.8,
        },
    ),
    "city": _mapping(
        families={"FRESH": 0.8, "WOODY": 0.7, "AMBERY": 0.5},
        subs={"Aromatic": 0.7, "Dry Woods": 0.5, "Soft Amber": 0.5},
        components={
            "Musk": 0.9,
            "Vetiver": 0.7,
            "Cedarwood": 0.5,
        },
        descriptors={
            "세련된": 1.2,
            "깨끗한": 0.8,
            "차분한": 0.8,
        },
    ),
    "office": _mapping(
        families={"FRESH": 1.2},
        subs={"Green": 0.8, "Citrus": 0.8},
        components={
            "Musk": 1.0,
            "Bergamot": 0.8,
            "Green tea": 0.7,
        },
        descriptors={
            "깨끗한": 1.2,
            "가벼운": 0.9,
            "단정한": 0.9,
        },
    ),
    "bedroom": _mapping(
        families={"AMBERY": 0.8, "FLORAL": 0.8},
        subs={"Soft Amber": 0.8, "Soft Floral": 0.8},
        components={
            "Musk": 1.0,
            "Vanilla": 0.7,
            "Iris": 0.8,
        },
        descriptors={
            "포근한": 1.0,
            "부드러운": 1.0,
            "차분한": 0.8,
        },
    ),

    # -----------------------------------------------------
    # moods / raw keywords
    # -----------------------------------------------------
    "modern": _mapping(
        families={"FRESH": 1.1, "FLORAL": 0.7, "WOODY": 0.4},
        subs={"Green": 0.8, "Citrus": 0.7, "Soft Floral": 0.6},
        components={
            "Musk": 1.0,
            "Bergamot": 0.8,
            "Green Note": 0.7,
            "Iris": 0.5,
        },
        descriptors={
            "세련된": 1.4,
            "깨끗한": 1.0,
            "상쾌한": 0.8,
            "가벼운": 0.7,
        },
    ),
    "fashion": _mapping(
        families={"FLORAL": 1.2, "FRESH": 1.0, "AMBERY": 0.4},
        subs={"Soft Floral": 1.0, "Floral": 0.8, "Green": 0.6},
        components={
            "Musk": 1.1,
            "Iris": 0.9,
            "Peony": 0.9,
            "Absolute Rose": 0.8,
            "Bergamot": 0.6,
        },
        descriptors={
            "세련된": 1.3,
            "우아한": 1.1,
            "부드러운": 0.8,
            "밝은": 0.6,
        },
    ),
    "style": _mapping(
        families={"FLORAL": 0.8, "FRESH": 0.8},
        subs={"Soft Floral": 0.7, "Green": 0.5},
        components={
            "Musk": 0.8,
            "Iris": 0.7,
            "Bergamot": 0.5,
        },
        descriptors={
            "세련된": 1.1,
            "우아한": 0.7,
            "깨끗한": 0.5,
        },
    ),
    "feminine": _mapping(
        families={"FLORAL": 1.6, "FRESH": 0.8},
        subs={"Floral": 1.2, "Soft Floral": 1.2, "Fruity": 0.5},
        components={
            "Peony": 1.2,
            "Absolute Rose": 1.2,
            "Absolute Jasmin": 0.9,
            "Iris": 0.8,
            "Musk": 0.7,
        },
        descriptors={
            "우아한": 1.4,
            "섬세한": 1.1,
            "부드러운": 1.0,
            "감미로운": 0.8,
        },
    ),
    "elegant": _mapping(
        families={"FLORAL": 1.4, "AMBERY": 0.8, "FRESH": 0.6},
        subs={"Soft Floral": 1.2, "Floral": 0.9, "Soft Amber": 0.6},
        components={
            "Iris": 1.2,
            "Musk": 1.1,
            "Absolute Rose": 1.0,
            "Peony": 0.8,
        },
        descriptors={
            "우아한": 1.8,
            "부드러운": 1.0,
            "섬세한": 0.9,
            "매력적인": 0.7,
        },
    ),
    "luxury": _mapping(
        families={"AMBERY": 1.0, "FLORAL": 0.8, "WOODY": 0.5},
        subs={"Amber": 0.8, "Soft Amber": 0.7, "Floral Amber": 0.6},
        components={
            "Amber": 1.0,
            "Musk": 0.8,
            "Absolute Rose": 0.7,
            "Sandalwood": 0.6,
        },
        descriptors={
            "우아한": 1.1,
            "매력적인": 1.0,
            "세련된": 0.9,
            "깊은": 0.6,
        },
    ),
    "contemporary": _mapping(
        families={"FRESH": 1.0, "FLORAL": 0.7},
        subs={"Green": 0.7, "Citrus": 0.6, "Soft Floral": 0.5},
        components={
            "Musk": 0.9,
            "Bergamot": 0.8,
            "Green Note": 0.6,
        },
        descriptors={
            "세련된": 1.2,
            "깨끗한": 0.8,
            "가벼운": 0.6,
        },
    ),
    "warm": _mapping(
        families={"AMBERY": 1.6, "WOODY": 1.0},
        subs={"Amber": 1.4, "Soft Amber": 1.2, "Woody Amber": 0.8},
        components={
            "Amber": 1.5,
            "Vanilla": 1.2,
            "Sandalwood": 1.0,
            "Tobacco": 0.7,
        },
        descriptors={
            "따뜻한": 2.0,
            "포근한": 1.3,
            "깊은": 0.8,
        },
    ),
    "dark": _mapping(
        families={"WOODY": 1.5, "AMBERY": 1.2},
        subs={"Dry Woods": 1.5, "Soft Amber": 0.8},
        components={
            "Leather": 1.3,
            "Incense": 1.3,
            "Oud": 1.2,
            "Patchouli": 1.0,
        },
        descriptors={
            "스모키한": 1.4,
            "묵직한": 1.5,
            "신비로운": 1.2,
            "깊은": 1.2,
        },
    ),
    "luxurious": _mapping(
        families={"AMBERY": 1.2, "WOODY": 0.9, "FLORAL": 0.9},
        subs={"Amber": 1.0, "Woody Amber": 0.8, "Floral Amber": 0.8},
        components={
            "Amber": 1.1,
            "Sandalwood": 0.9,
            "Absolute Rose": 0.9,
            "Musk": 0.8,
        },
        descriptors={
            "우아한": 1.2,
            "감각적인": 1.0,
            "매력적인": 1.1,
            "깊은": 0.7,
        },
    ),
    "clean": _mapping(
        families={"FRESH": 1.8},
        subs={"Green": 1.0, "Citrus": 1.0, "Water": 0.8},
        components={
            "Musk": 1.3,
            "Bergamot": 1.0,
            "Green tea": 0.9,
            "Lemon": 0.8,
        },
        descriptors={
            "깨끗한": 2.0,
            "청량한": 1.2,
            "상쾌한": 1.0,
            "가벼운": 1.0,
        },
    ),
    "fresh": _mapping(
        families={"FRESH": 2.0},
        subs={"Citrus": 1.3, "Green": 1.1, "Water": 0.8},
        components={
            "Bergamot": 1.2,
            "Lemon": 1.1,
            "Mandarin": 0.9,
            "Green Note": 0.9,
            "Marine": 0.7,
        },
        descriptors={
            "상쾌한": 1.8,
            "시원한": 1.2,
            "산뜻한": 1.2,
            "가벼운": 1.0,
        },
    ),
    "romantic": _mapping(
        families={"FLORAL": 1.8},
        subs={"Floral": 1.4, "Soft Floral": 1.2},
        components={
            "Absolute Rose": 1.4,
            "Peony": 1.1,
            "Absolute Jasmin": 1.0,
            "Iris": 0.8,
        },
        descriptors={
            "우아한": 1.4,
            "섬세한": 1.2,
            "부드러운": 1.2,
            "감미로운": 1.0,
        },
    ),
    "sensual": _mapping(
        families={"AMBERY": 1.5, "WOODY": 1.0, "FLORAL": 0.8},
        subs={"Floral Amber": 1.2, "Woody Amber": 1.0, "Dry Woods": 0.7},
        components={
            "Amber": 1.4,
            "Leather": 0.9,
            "Vanilla": 1.0,
            "Patchouli": 0.8,
            "Absolute Rose": 0.8,
        },
        descriptors={
            "감각적인": 1.8,
            "따뜻한": 1.1,
            "깊은": 0.9,
            "매력적인": 1.2,
        },
    ),
    "soft": _mapping(
        families={"FLORAL": 1.0, "AMBERY": 1.0},
        subs={"Soft Floral": 1.3, "Soft Amber": 1.2},
        components={
            "Musk": 1.2,
            "Iris": 1.0,
            "Vanilla": 0.8,
            "Lily": 0.7,
        },
        descriptors={
            "부드러운": 2.0,
            "포근한": 1.0,
            "온화한": 0.9,
        },
    ),
    "natural": _mapping(
        families={"FRESH": 1.3, "WOODY": 1.0},
        subs={"Green": 1.2, "Mossy Woods": 0.9, "Aromatic": 0.8},
        components={
            "Green Note": 1.2,
            "Oakmoss": 0.9,
            "Vetiver": 0.8,
            "Basil": 0.8,
        },
        descriptors={
            "신선한": 1.3,
            "청정한": 1.2,
            "상쾌한": 1.0,
        },
    ),
    "urban": _mapping(
        families={"FRESH": 0.8, "WOODY": 0.7, "AMBERY": 0.5},
        subs={"Aromatic": 0.8, "Soft Amber": 0.5},
        components={
            "Musk": 1.0,
            "Vetiver": 0.7,
            "Bergamot": 0.6,
        },
        descriptors={
            "도시적인": 1.3,
            "세련된": 1.1,
            "깨끗한": 0.7,
        },
    ),
    "calm": _mapping(
        families={"AMBERY": 0.8, "FRESH": 0.8, "WOODY": 0.6},
        subs={"Soft Amber": 0.8, "Green": 0.7, "Woods": 0.5},
        components={
            "Musk": 1.0,
            "Sandalwood": 0.7,
            "Green tea": 0.7,
        },
        descriptors={
            "차분한": 1.8,
            "부드러운": 0.9,
            "안정감": 0.8,
        },
    ),
    "bright": _mapping(
        families={"FRESH": 1.4, "FLORAL": 0.8},
        subs={"Citrus": 1.2, "Fruity": 0.8, "Floral": 0.7},
        components={
            "Bergamot": 1.1,
            "Lemon": 1.0,
            "Orange": 1.0,
            "Peony": 0.6,
        },
        descriptors={
            "밝은": 1.8,
            "상큼한": 1.2,
            "가벼운": 1.0,
        },
    ),
    "cozy": _mapping(
        families={"AMBERY": 1.3, "WOODY": 0.6},
        subs={"Soft Amber": 1.3, "Amber": 0.7},
        components={
            "Vanilla": 1.3,
            "Musk": 1.0,
            "Amber": 1.0,
            "Sandalwood": 0.6,
        },
        descriptors={
            "포근한": 1.8,
            "따뜻한": 1.2,
            "부드러운": 1.0,
        },
    ),

    # -----------------------------------------------------
    # season / time
    # -----------------------------------------------------
    "spring": _mapping(
        families={"FLORAL": 1.4, "FRESH": 1.3},
        subs={"Floral": 1.1, "Soft Floral": 1.0, "Green": 1.0, "Citrus": 0.6},
        components={
            "Peony": 1.2,
            "Absolute Rose": 1.0,
            "Green Note": 1.0,
            "Bergamot": 0.8,
            "Musk": 0.5,
        },
        descriptors={
            "신선한": 1.2,
            "상쾌한": 1.1,
            "밝은": 1.0,
            "부드러운": 0.8,
        },
    ),
    "summer": _mapping(
        families={"FRESH": 1.6},
        subs={"Citrus": 1.2, "Water": 1.0, "Green": 0.8},
        components={
            "Bergamot": 1.1,
            "Lemon": 1.0,
            "Marine": 0.9,
            "Green Note": 0.8,
        },
        descriptors={
            "시원한": 1.3,
            "상쾌한": 1.3,
            "가벼운": 1.0,
        },
    ),
    "autumn": _mapping(
        families={"WOODY": 1.2, "AMBERY": 1.0},
        subs={"Woods": 1.0, "Woody Amber": 0.9, "Amber": 0.8},
        components={
            "Sandalwood": 1.0,
            "Amber": 1.0,
            "Patchouli": 0.8,
            "Cedarwood": 0.8,
        },
        descriptors={
            "따뜻한": 1.0,
            "깊은": 0.9,
            "차분한": 0.8,
        },
    ),
    "winter": _mapping(
        families={"AMBERY": 1.4, "WOODY": 1.0},
        subs={"Amber": 1.2, "Soft Amber": 1.0, "Dry Woods": 0.8},
        components={
            "Amber": 1.3,
            "Vanilla": 1.0,
            "Incense": 0.8,
            "Sandalwood": 0.7,
        },
        descriptors={
            "따뜻한": 1.3,
            "포근한": 1.1,
            "묵직한": 0.8,
        },
    ),
    "morning": _mapping(
        families={"FRESH": 1.0},
        subs={"Citrus": 0.9, "Green": 0.8},
        components={
            "Bergamot": 0.9,
            "Lemon": 0.8,
            "Green Note": 0.7,
        },
        descriptors={
            "상쾌한": 1.0,
            "깨끗한": 0.8,
            "가벼운": 0.8,
        },
    ),
    "afternoon": _mapping(
        families={"FRESH": 0.9, "FLORAL": 0.7},
        subs={"Green": 0.7, "Soft Floral": 0.6, "Citrus": 0.5},
        components={
            "Bergamot": 0.7,
            "Green Note": 0.7,
            "Musk": 0.6,
            "Peony": 0.5,
        },
        descriptors={
            "밝은": 0.8,
            "상쾌한": 0.8,
            "가벼운": 0.7,
        },
    ),
    "evening": _mapping(
        families={"AMBERY": 1.0, "WOODY": 0.8},
        subs={"Soft Amber": 0.9, "Woody Amber": 0.7},
        components={
            "Amber": 1.0,
            "Musk": 0.8,
            "Sandalwood": 0.7,
        },
        descriptors={
            "따뜻한": 0.9,
            "차분한": 0.8,
            "깊은": 0.7,
        },
    ),
    "night": _mapping(
        families={"WOODY": 1.2, "AMBERY": 1.2},
        subs={"Dry Woods": 1.0, "Amber": 1.0, "Soft Amber": 0.8},
        components={
            "Leather": 0.9,
            "Amber": 1.0,
            "Incense": 0.9,
            "Musk": 0.8,
        },
        descriptors={
            "깊은": 1.1,
            "감각적인": 0.9,
            "묵직한": 0.8,
            "신비로운": 0.8,
        },
    ),
}


KOREAN_VISUAL_TRIGGERS: dict[str, str] = {
    "검정": "black",
    "검은": "black",
    "갈색": "brown",
    "브라운": "brown",
    "금색": "gold",
    "골드": "gold",
    "초록": "green",
    "녹색": "green",
    "파랑": "blue",
    "파란": "blue",
    "청록": "green",
    "흰색": "white",
    "하얀": "white",
    "분홍": "pink",
    "핑크": "pink",
    "빨강": "red",
    "빨간": "red",
    "베이지": "beige",
    "회색": "gray",
    "그레이": "gray",

    "드레스": "dress",
    "핸드백": "handbag",
    "가방": "handbag",
    "손가방": "handbag",
    "목걸이": "necklace",
    "주얼리": "necklace",

    "나무": "wood",
    "목재": "wood",
    "가죽": "leather",
    "레더": "leather",
    "꽃": "flower",
    "바다": "ocean",
    "커피": "coffee",
    "캔들": "candle",
    "책": "book",
    "천": "fabric",
    "패브릭": "fabric",

    "숲": "forest",
    "해변": "beach",
    "카페": "cafe",
    "바 ": "bar",
    "술집": "bar",
    "정원": "garden",
    "비": "rainy street",
    "도시": "city",
    "방": "room",
    "실내": "indoor",
    "오피스": "office",
    "사무실": "office",

    "현대적인": "modern",
    "모던": "modern",
    "패션": "fashion",
    "스타일": "style",
    "여성": "feminine",
    "여성적": "feminine",
    "우아": "elegant",
    "엘레강스": "elegant",
    "동시대": "contemporary",
    "세련": "contemporary",
    "럭셔리": "luxury",
    "고급": "luxurious",

    "따뜻": "warm",
    "어두": "dark",
    "깨끗": "clean",
    "상쾌": "fresh",
    "신선": "fresh",
    "로맨틱": "romantic",
    "감각": "sensual",
    "관능": "sensual",
    "자연": "natural",
    "차분": "calm",
    "포근": "cozy",
    "밝": "bright",
    "부드": "soft",

    "봄": "spring",
    "여름": "summer",
    "가을": "autumn",
    "겨울": "winter",
    "오전": "morning",
    "아침": "morning",
    "오후": "afternoon",
    "저녁": "evening",
    "밤": "night",
}


def _collect_source_keywords(image_keywords: dict[str, Any]) -> list[str]:
    source: list[str] = []

    for key in ["colors", "objects", "scene", "mood", "season", "time", "raw_keywords"]:
        values = image_keywords.get(key, [])

        if isinstance(values, list):
            source.extend([str(v).strip().lower() for v in values if str(v).strip()])
        elif isinstance(values, str) and values.strip():
            source.append(values.strip().lower())

    summary = str(image_keywords.get("visual_summary", "")).strip().lower()
    if summary:
        source.append(summary)

    return source


def _detect_triggers(source_keywords: list[str]) -> list[str]:
    joined = " ".join(source_keywords).lower()
    detected: list[str] = []

    for trigger in VISUAL_TO_FRAGRANCE_RULES:
        if trigger in joined:
            detected.append(trigger)

    for ko_trigger, en_trigger in KOREAN_VISUAL_TRIGGERS.items():
        if ko_trigger in joined:
            detected.append(en_trigger)

    return detected


def _add_scores(
    target: defaultdict[str, float],
    score_map: dict[str, float],
    multiplier: float = 1.0,
    normalize_component: bool = False,
) -> None:
    for key, score in score_map.items():
        normalized_key = normalize_note_keyword(key) if normalize_component else key
        if not normalized_key:
            continue
        target[normalized_key] += score * multiplier


def _subtract_score(
    target: defaultdict[str, float],
    key: str,
    amount: float,
    normalize_component: bool = False,
) -> None:
    normalized_key = normalize_note_keyword(key) if normalize_component else key
    if not normalized_key:
        return
    target[normalized_key] = max(0.0, target.get(normalized_key, 0.0) - amount)


def _build_component_ko_scores(component_scores: dict[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}

    for component, score in component_scores.items():
        kor = to_korean_note(component)
        result[kor] = result.get(kor, 0.0) + score

    return result


def _rank_scores(
    score_dict: dict[str, float],
    top_n: int | None = None,
) -> list[dict[str, float | str]]:
    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)

    if top_n:
        ranked = ranked[:top_n]

    return [
        {
            "name": name,
            "score": round(score, 3),
        }
        for name, score in ranked
        if score > 0
    ]


def _names_only(ranked_items: list[dict[str, float | str]]) -> list[str]:
    return [str(item["name"]) for item in ranked_items]


def _terms_for_query(
    ranked_items: list[dict[str, float | str]],
) -> list[str]:
    """
    query_text에는 각 키워드를 한 번씩만 넣는다.
    점수 정보는 scores 필드에 따로 보관한다.
    """
    return [
        str(item["name"]).strip()
        for item in ranked_items
        if str(item.get("name", "")).strip()
    ]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        cleaned = str(item).strip()
        if not cleaned:
            continue

        if cleaned in seen:
            continue

        seen.add(cleaned)
        result.append(cleaned)

    return result


def _dedupe_query_parts(parts: list[str]) -> list[str]:
    """
    query_text 내부 중복 제거.

    prefix가 붙어 있기 때문에:
    - visual_color:green
    - fragrance_sub:Green

    위 둘은 서로 다른 의미로 유지된다.
    """

    seen = set()
    deduped = []

    for part in parts:
        cleaned = str(part).strip()
        if not cleaned:
            continue

        key = cleaned.lower()
        if key in seen:
            continue

        seen.add(key)
        deduped.append(cleaned)

    return deduped


def _add_prefixed_terms(
    parts: list[str],
    prefix: str,
    values: list[str] | str | None,
) -> None:
    if values is None:
        return

    if isinstance(values, str):
        values = [values]

    if not isinstance(values, list):
        return

    for value in values:
        cleaned = str(value).strip()
        if cleaned:
            parts.append(f"{prefix}:{cleaned}")


def _has_any(source_keywords: list[str], candidates: list[str]) -> bool:
    joined = " ".join(source_keywords).lower()
    return any(candidate.lower() in joined for candidate in candidates)


def _apply_context_adjustments(
    source_keywords: list[str],
    matched_triggers: list[str],
    family_scores: defaultdict[str, float],
    sub_scores: defaultdict[str, float],
    component_scores: defaultdict[str, float],
    descriptor_scores: defaultdict[str, float],
) -> None:
    """
    단순 trigger 매핑 이후, 이미지 전체 문맥에 따라 점수를 보정한다.

    핵심:
    - black 단독은 무거운 향으로 과도하게 해석하지 않는다.
    - black + dark/night/bar/leather 조합이면 그때 Dry Woods / Leather / Incense를 강화한다.
    - fashion/feminine/elegant/spring/afternoon 문맥이면 Fresh/Floral을 강화하고 무거운 노트를 낮춘다.
    """

    fashion_context = _has_any(
        source_keywords,
        [
            "fashion",
            "style",
            "feminine",
            "elegant",
            "contemporary",
            "dress",
            "handbag",
            "necklace",
            "spring",
            "afternoon",
            "modern",
            "패션",
            "여성",
            "우아",
            "드레스",
            "봄",
            "오후",
        ],
    )

    dark_context = _has_any(
        source_keywords,
        [
            "dark",
            "night",
            "bar",
            "leather",
            "black leather",
            "smoke",
            "candle",
            "어두",
            "밤",
            "가죽",
            "술집",
            "스모키",
        ],
    )

    if "black" in matched_triggers and not dark_context:
        _subtract_score(component_scores, "Leather", 0.6, normalize_component=True)
        _subtract_score(component_scores, "Incense", 0.5, normalize_component=True)
        _subtract_score(component_scores, "Oud", 0.5, normalize_component=True)
        _subtract_score(sub_scores, "Dry Woods", 0.5)
        _subtract_score(descriptor_scores, "스모키한", 0.5)
        _subtract_score(descriptor_scores, "묵직한", 0.4)

        descriptor_scores["세련된"] += 0.8
        descriptor_scores["차분한"] += 0.4
        component_scores[normalize_note_keyword("Musk")] += 0.5

    if "black" in matched_triggers and dark_context:
        family_scores["WOODY"] += 0.8
        family_scores["AMBERY"] += 0.6
        sub_scores["Dry Woods"] += 0.9
        component_scores[normalize_note_keyword("Leather")] += 1.0
        component_scores[normalize_note_keyword("Incense")] += 0.8
        component_scores[normalize_note_keyword("Oud")] += 0.8
        descriptor_scores["스모키한"] += 0.9
        descriptor_scores["묵직한"] += 0.8
        descriptor_scores["깊은"] += 0.8

    if fashion_context:
        family_scores["FRESH"] += 1.2
        family_scores["FLORAL"] += 1.3

        sub_scores["Soft Floral"] += 1.0
        sub_scores["Floral"] += 0.8
        sub_scores["Green"] += 0.7
        sub_scores["Citrus"] += 0.5

        component_scores[normalize_note_keyword("Musk")] += 1.0
        component_scores[normalize_note_keyword("Iris")] += 0.9
        component_scores[normalize_note_keyword("Peony")] += 0.9
        component_scores[normalize_note_keyword("Absolute Rose")] += 0.8
        component_scores[normalize_note_keyword("Bergamot")] += 0.7
        component_scores[normalize_note_keyword("Green Note")] += 0.6

        descriptor_scores["우아한"] += 1.2
        descriptor_scores["세련된"] += 1.1
        descriptor_scores["부드러운"] += 0.9
        descriptor_scores["밝은"] += 0.8
        descriptor_scores["상쾌한"] += 0.7

        if not dark_context:
            _subtract_score(component_scores, "Leather", 0.5, normalize_component=True)
            _subtract_score(component_scores, "Incense", 0.5, normalize_component=True)
            _subtract_score(component_scores, "Oud", 0.5, normalize_component=True)
            _subtract_score(sub_scores, "Dry Woods", 0.5)
            _subtract_score(descriptor_scores, "스모키한", 0.5)
            _subtract_score(descriptor_scores, "묵직한", 0.4)


def _build_query_sections(
    image_keywords: dict[str, Any],
    fragrance_keywords: dict[str, Any],
) -> dict[str, list[str] | str]:
    scores = fragrance_keywords.get("scores", {})

    visual_summary = image_keywords.get("visual_summary", "")
    if not isinstance(visual_summary, str):
        visual_summary = str(visual_summary)

    query_sections: dict[str, list[str] | str] = {
        "visual_summary": visual_summary.strip(),
        "visual_colors": _dedupe_preserve_order(
            [str(v).strip() for v in image_keywords.get("colors", []) if str(v).strip()]
        ),
        "visual_objects": _dedupe_preserve_order(
            [str(v).strip() for v in image_keywords.get("objects", []) if str(v).strip()]
        ),
        "visual_scene": _dedupe_preserve_order(
            [str(v).strip() for v in image_keywords.get("scene", []) if str(v).strip()]
        ),
        "visual_mood": _dedupe_preserve_order(
            [str(v).strip() for v in image_keywords.get("mood", []) if str(v).strip()]
        ),
        "visual_season": _dedupe_preserve_order(
            [str(v).strip() for v in image_keywords.get("season", []) if str(v).strip()]
        ),
        "visual_time": _dedupe_preserve_order(
            [str(v).strip() for v in image_keywords.get("time", []) if str(v).strip()]
        ),
        "visual_raw_keywords": _dedupe_preserve_order(
            [str(v).strip() for v in image_keywords.get("raw_keywords", []) if str(v).strip()]
        ),

        "fragrance_families": _terms_for_query(scores.get("families", [])),
        "fragrance_subs": _terms_for_query(scores.get("subs", [])),
        "fragrance_components": _terms_for_query(scores.get("components", [])),
        "fragrance_components_ko": _terms_for_query(scores.get("components_ko", [])),
        "fragrance_descriptors": _terms_for_query(scores.get("descriptors", [])),
    }

    return query_sections


def build_image_query_text(
    image_keywords: dict[str, Any],
    fragrance_keywords: dict[str, Any],
) -> str:
    """
    RAG 검색에 사용할 query_text를 생성한다.

    핵심:
    - visual_color:green
    - fragrance_sub:Green

    같은 단어라도 의미 영역이 다르면 둘 다 유지한다.
    """

    sections = _build_query_sections(image_keywords, fragrance_keywords)
    parts: list[str] = []

    visual_summary = sections.get("visual_summary", "")
    if isinstance(visual_summary, str) and visual_summary.strip():
        parts.append(f"visual_summary:{visual_summary.strip()}")

    _add_prefixed_terms(parts, "visual_color", sections.get("visual_colors", []))
    _add_prefixed_terms(parts, "visual_object", sections.get("visual_objects", []))
    _add_prefixed_terms(parts, "visual_scene", sections.get("visual_scene", []))
    _add_prefixed_terms(parts, "visual_mood", sections.get("visual_mood", []))
    _add_prefixed_terms(parts, "visual_season", sections.get("visual_season", []))
    _add_prefixed_terms(parts, "visual_time", sections.get("visual_time", []))
    _add_prefixed_terms(parts, "visual_keyword", sections.get("visual_raw_keywords", []))

    _add_prefixed_terms(parts, "fragrance_family", sections.get("fragrance_families", []))
    _add_prefixed_terms(parts, "fragrance_sub", sections.get("fragrance_subs", []))
    _add_prefixed_terms(parts, "fragrance_note", sections.get("fragrance_components", []))
    _add_prefixed_terms(parts, "fragrance_note_ko", sections.get("fragrance_components_ko", []))
    _add_prefixed_terms(parts, "fragrance_descriptor", sections.get("fragrance_descriptors", []))

    return " ".join(_dedupe_query_parts(parts))


def _build_readable_query_text(
    image_keywords: dict[str, Any],
    fragrance_keywords: dict[str, Any],
) -> str:
    """
    사람이 읽기 좋은 확인용 query_text.

    RAG 검색에는 prefix가 붙은 query_text를 쓰고,
    화면 표시나 디버깅에는 readable_query_text를 쓰면 된다.
    """

    sections = _build_query_sections(image_keywords, fragrance_keywords)
    parts: list[str] = []

    visual_summary = sections.get("visual_summary", "")
    if isinstance(visual_summary, str) and visual_summary.strip():
        parts.append(visual_summary.strip())

    for key in [
        "visual_colors",
        "visual_objects",
        "visual_scene",
        "visual_mood",
        "visual_season",
        "visual_time",
        "visual_raw_keywords",
        "fragrance_families",
        "fragrance_subs",
        "fragrance_components",
        "fragrance_components_ko",
        "fragrance_descriptors",
    ]:
        values = sections.get(key, [])
        if isinstance(values, list):
            parts.extend(values)

    return " ".join(_dedupe_query_parts(parts))


def map_image_to_fragrance_keywords(image_keywords: dict[str, Any]) -> dict[str, Any]:
    source_keywords = _collect_source_keywords(image_keywords)
    matched_triggers = _detect_triggers(source_keywords)

    family_scores: defaultdict[str, float] = defaultdict(float)
    sub_scores: defaultdict[str, float] = defaultdict(float)
    component_scores: defaultdict[str, float] = defaultdict(float)
    descriptor_scores: defaultdict[str, float] = defaultdict(float)

    trigger_counter = Counter(matched_triggers)

    for trigger, count in trigger_counter.items():
        mapping = VISUAL_TO_FRAGRANCE_RULES.get(trigger)
        if not mapping:
            continue

        multiplier = min(1.0 + (count - 1) * 0.25, 1.5)

        _add_scores(family_scores, mapping.get("families", {}), multiplier)
        _add_scores(sub_scores, mapping.get("subs", {}), multiplier)
        _add_scores(
            component_scores,
            mapping.get("components", {}),
            multiplier,
            normalize_component=True,
        )
        _add_scores(descriptor_scores, mapping.get("descriptors", {}), multiplier)

    _apply_context_adjustments(
        source_keywords=source_keywords,
        matched_triggers=matched_triggers,
        family_scores=family_scores,
        sub_scores=sub_scores,
        component_scores=component_scores,
        descriptor_scores=descriptor_scores,
    )

    component_ko_scores = _build_component_ko_scores(component_scores)

    ranked_families = _rank_scores(family_scores, top_n=3)
    ranked_subs = _rank_scores(sub_scores, top_n=6)
    ranked_components = _rank_scores(component_scores, top_n=10)
    ranked_components_ko = _rank_scores(component_ko_scores, top_n=10)
    ranked_descriptors = _rank_scores(descriptor_scores, top_n=10)

    result: dict[str, Any] = {
        "matched_triggers": _dedupe_preserve_order(matched_triggers),

        "scores": {
            "families": ranked_families,
            "subs": ranked_subs,
            "components": ranked_components,
            "components_ko": ranked_components_ko,
            "descriptors": ranked_descriptors,
        },

        "fragrance_families": _names_only(ranked_families),
        "fragrance_subs": _names_only(ranked_subs),
        "components": _names_only(ranked_components),
        "components_ko": _names_only(ranked_components_ko),
        "descriptors": _names_only(ranked_descriptors),
    }

    result["query_sections"] = _build_query_sections(image_keywords, result)
    result["query_text"] = build_image_query_text(image_keywords, result)
    result["readable_query_text"] = _build_readable_query_text(image_keywords, result)

    return result


if __name__ == "__main__":
    import json

    sample = {
        "visual_summary": "청록색과 검은색 체크무늬 드레스를 입은 여성이 실내에서 포즈를 취하고 있습니다.",
        "colors": ["black", "green", "blue", "pink", "red"],
        "objects": ["handbag", "dress", "necklace"],
        "scene": ["indoor"],
        "mood": ["soft", "romantic"],
        "season": ["spring"],
        "time": ["afternoon"],
        "raw_keywords": ["체크무늬", "드레스", "여성", "실내", "포즈", "손가방", "목걸이"],
    }

    result = map_image_to_fragrance_keywords(sample)
    print(json.dumps(result, ensure_ascii=False, indent=2))