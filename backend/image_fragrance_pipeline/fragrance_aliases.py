"""
fragrance_aliases.py
--------------------
이미지 분석 결과와 크롤링 데이터의 표기를 같은 향료 키워드로 정규화하는 모듈.
예: 레더/가죽향/leather -> Leather
"""

from __future__ import annotations


def _norm(text: str) -> str:
    return str(text or "").strip().lower()


NOTE_ALIAS_MAP: dict[str, str] = {
    # Woody / dry
    "leather": "Leather",
    "leather accord": "Leather",
    "suede": "Leather",
    "레더": "Leather",
    "가죽": "Leather",
    "가죽향": "Leather",

    "tobacco": "Tobacco",
    "tabaco": "Tobacco",  # taxonomy typo fallback
    "토바코": "Tobacco",
    "담배": "Tobacco",
    "담배향": "Tobacco",

    "oud": "Oud",
    "agarwood": "Oud",
    "우드": "Oud",
    "오우드": "Oud",

    "cedarwood": "Cedarwood",
    "cedar": "Cedarwood",
    "cedar wood": "Cedarwood",
    "시더우드": "Cedarwood",
    "삼나무": "Cedarwood",

    "sandalwood": "Sandalwood",
    "sandal wood": "Sandalwood",
    "샌달우드": "Sandalwood",

    "vetiver": "Vetiver",
    "베티버": "Vetiver",

    "patchouli": "Patchouli",
    "패출리": "Patchouli",
    "파출리": "Patchouli",

    "oakmoss": "Oakmoss",
    "oak moss": "Oakmoss",
    "오크모스": "Oakmoss",

    "pine": "Pine tree",
    "pine tree": "Pine tree",
    "소나무": "Pine tree",

    "wood": "Wood Scent",
    "woods": "Wood Scent",
    "wood scent": "Wood Scent",
    "woody": "Wood Scent",
    "나무": "Wood Scent",
    "나무향": "Wood Scent",
    "우디": "Wood Scent",

    "earthy": "Earthy",
    "earth": "Earthy",
    "soil": "Earthy",
    "흙": "Earthy",
    "흙냄새": "Earthy",
    "어시향": "Earthy",

    # Ambery / warm
    "amber": "Amber",
    "앰버": "Amber",

    "musk": "Musk",
    "머스크": "Musk",

    "vanilla": "Vanilla",
    "바닐라": "Vanilla",

    "incense": "Incense",
    "인센스": "Incense",
    "smoke": "Incense",
    "smoky": "Incense",
    "연기": "Incense",
    "스모키": "Incense",
    "스모키한": "Incense",

    "cardamom": "Cardamom",
    "카다멈": "Cardamom",

    "pepper": "Pepper",
    "black pepper": "Pepper",
    "pink pepper": "Pepper",
    "후추": "Pepper",
    "페퍼": "Pepper",

    "spice": "Spicy",
    "spicy": "Spicy",
    "향신료": "Spicy",
    "스파이시": "Spicy",
    "스파이시한": "Spicy",

    "clove": "Clove Bud",
    "clove bud": "Clove Bud",
    "클로브": "Clove Bud",
    "클로브 버드": "Clove Bud",

    # Floral
    "rose": "Absolute Rose",
    "absolute rose": "Absolute Rose",
    "로즈": "Absolute Rose",
    "장미": "Absolute Rose",
    "앱솔루트 로즈": "Absolute Rose",

    "jasmine": "Absolute Jasmin",
    "jasmin": "Absolute Jasmin",
    "absolute jasmine": "Absolute Jasmin",
    "absolute jasmin": "Absolute Jasmin",
    "자스민": "Absolute Jasmin",
    "앱솔루트 자스민": "Absolute Jasmin",

    "peony": "Peony",
    "피오니": "Peony",

    "lily": "Lily",
    "백합": "Lily",

    "iris": "Iris",
    "아이리스": "Iris",

    "orange flower": "Absolute Orange Flower",
    "orange blossom": "Absolute Orange Flower",
    "neroli": "Absolute Orange Flower",
    "오렌지꽃": "Absolute Orange Flower",
    "오렌지 블라썸": "Absolute Orange Flower",
    "네롤리": "Absolute Orange Flower",
    "앱솔루트 오렌지꽃": "Absolute Orange Flower",

    # Fresh / citrus / green / water
    "bergamot": "Bergamot",
    "베르가못": "Bergamot",

    "lemon": "Lemon",
    "레몬": "Lemon",

    "mandarin": "Mandarin",
    "만다린": "Mandarin",

    "orange": "Orange",
    "오렌지": "Orange",

    "citrus": "Citrus",
    "시트러스": "Citrus",

    "lavender": "Lavender",
    "라벤더": "Lavender",

    "rosemary": "Rosemary",
    "로즈마리": "Rosemary",

    "basil": "Basil",
    "바질": "Basil",

    "herb": "Herb",
    "herbal": "Herb",
    "허브": "Herb",

    "green": "Green Note",
    "green note": "Green Note",
    "그린": "Green Note",
    "그린노트": "Green Note",
    "grass": "Green Note",
    "leaf": "Green Note",
    "leaves": "Green Note",
    "풀": "Green Note",
    "잎": "Green Note",

    "green tea": "Green tea",
    "greentea": "Green tea",
    "그린티": "Green tea",
    "녹차": "Green tea",

    "marine": "Marine",
    "sea": "Marine",
    "ocean": "Marine",
    "beach": "Marine",
    "마린": "Marine",
    "바다": "Marine",
    "해변": "Marine",

    "ozonic": "Ozonic",
    "rain": "Ozonic",
    "rainy": "Ozonic",
    "오조닉": "Ozonic",
    "비": "Ozonic",
    "비오는": "Ozonic",

    "berry": "Berry, Apple, Peach",
    "berries": "Berry, Apple, Peach",
    "apple": "Berry, Apple, Peach",
    "peach": "Berry, Apple, Peach",
    "fruit": "Berry, Apple, Peach",
    "fruity": "Berry, Apple, Peach",
    "베리": "Berry, Apple, Peach",
    "사과": "Berry, Apple, Peach",
    "복숭아": "Berry, Apple, Peach",
    "과일": "Berry, Apple, Peach",

    "tropical fruit": "Tropical Fruit",
    "tropical": "Tropical Fruit",
    "열대과일": "Tropical Fruit",
}


CANONICAL_TO_KO: dict[str, str] = {
    "Leather": "레더",
    "Tobacco": "토바코",
    "Oud": "우드",
    "Cedarwood": "시더우드",
    "Sandalwood": "샌달우드",
    "Vetiver": "베티버",
    "Patchouli": "패출리",
    "Oakmoss": "오크모스",
    "Pine tree": "소나무",
    "Wood Scent": "나무향",
    "Earthy": "어시향",
    "Amber": "앰버",
    "Musk": "머스크",
    "Vanilla": "바닐라",
    "Incense": "인센스",
    "Cardamom": "카다멈",
    "Pepper": "후추",
    "Spicy": "향신료",
    "Clove Bud": "클로브 버드",
    "Absolute Rose": "앱솔루트 로즈",
    "Absolute Jasmin": "앱솔루트 자스민",
    "Peony": "피오니",
    "Lily": "백합",
    "Iris": "아이리스",
    "Absolute Orange Flower": "앱솔루트 오렌지꽃",
    "Bergamot": "베르가못",
    "Lemon": "레몬",
    "Mandarin": "만다린",
    "Orange": "오렌지",
    "Citrus": "시트러스",
    "Lavender": "라벤더",
    "Rosemary": "로즈마리",
    "Basil": "바질",
    "Herb": "허브",
    "Green Note": "그린노트",
    "Green tea": "그린티",
    "Marine": "마린",
    "Ozonic": "오조닉",
    "Berry, Apple, Peach": "베리, 사과, 복숭아",
    "Tropical Fruit": "열대과일",
}


def normalize_note_keyword(keyword: str) -> str:
    """향료 키워드를 canonical English name으로 정규화한다."""
    key = _norm(keyword)
    if not key:
        return ""
    return NOTE_ALIAS_MAP.get(key, str(keyword).strip())


def to_korean_note(canonical_name: str) -> str:
    """canonical English name을 한국어 대표 표기로 변환한다."""
    return CANONICAL_TO_KO.get(canonical_name, canonical_name)
