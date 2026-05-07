"""
Tom Ford & Diptyque Fragrance Crawler FIXED (Scrapling + DeepL)
=========================================================

목표 출력 형식:
{
  "country": "US",
  "korean_name": "...",
  "english_name": "...",
  "product_type": "Eau de Parfum",
  "product_url": "https://...",
  "regular_price": "$250.00",
  "image_url": "https://...",
  "ingredients": "...",
  "key_ingredients": ["베르가못", "네롤리"]
}

설치:
    pip install -r requirements.txt
    scrapling install

.env:
    DEEPL_API_KEY=your_deepl_api_key_here
    DEEPL_API_URL=https://api-free.deepl.com/v2/translate

실행:
    python tomford_diptyque_spider.py
    python tomford_diptyque_spider.py tomford
    python tomford_diptyque_spider.py diptyque
    python tomford_diptyque_spider.py test https://www.example.com/product-url

주의:
- Tom Ford / Diptyque는 Creed와 HTML 구조가 다릅니다.
- 이 코드는 JSON-LD, script 내 product JSON, 링크/카드 텍스트를 순서대로 시도하는 범용 파서입니다.
- 사이트 구조가 바뀌면 CSS selector 또는 script 정규식 보정이 필요합니다.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin, urlparse

import requests
from dotenv import load_dotenv
from scrapling.spiders import Spider, Response, Request

load_dotenv()

# ============================================================
# DeepL 설정
# ============================================================

DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "").strip()
DEEPL_API_URL = os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate").strip()

if DEEPL_API_KEY:
    print(f"[DeepL] API Key 로드됨: {DEEPL_API_KEY[:6]}...{DEEPL_API_KEY[-4:]}")
else:
    print("[DeepL] ⚠ DEEPL_API_KEY 없음 → 한국어 필드는 원문/매핑 fallback으로 저장")

_translation_cache: dict[tuple[str, str, str], str] = {}

FRAGRANCE_TERM_MAP: dict[str, str] = {
    "Incense": "인센스", "Vetiver": "베티버", "Orris": "오리스", "Oud": "우드",
    "Ambroxan": "암브록산", "Amber": "앰버", "Musk": "머스크", "Patchouli": "패출리",
    "Bergamot": "베르가못", "Cedarwood": "시더우드", "Sandalwood": "샌달우드",
    "Tonka Bean": "통카빈", "Neroli": "네롤리", "Ylang-Ylang": "일랑일랑",
    "Galbanum": "갈바넘", "Heliotrope": "헬리오트로프", "Oakmoss": "오크모스",
    "Oak Moss": "오크모스", "Petitgrain": "쁘띠그레인", "Tuberose": "튜베로즈",
    "Osmanthus": "오스만투스", "Peach": "피치", "Rose": "로즈", "Lime": "라임",
    "Jasmine": "자스민", "Lily": "릴리", "Orchid": "오키드", "Violet": "바이올렛",
    "Vanilla": "바닐라", "Pepper": "페퍼", "Sage": "세이지", "Leather": "레더",
    "Tobacco": "토바코", "Blackcurrant": "블랙커런트", "Saffron": "사프란",
    "Cardamom": "카다멈", "Ginger": "진저", "Lemon": "레몬", "Mandarin": "만다린",
    "Orange": "오렌지", "Grapefruit": "그레이프프루트", "Pineapple": "파인애플",
    "Coconut": "코코넛", "Lavender": "라벤더", "Raspberry": "라즈베리",
    "Greens": "그린", "Cinnamon": "시나몬", "Iris": "아이리스", "Fig": "무화과",
    "Fig Tree": "무화과나무", "Cypress": "사이프러스", "Juniper": "주니퍼",
    "Myrrh": "미르", "Frankincense": "프랑킨센스", "Labdanum": "랍다넘",
    "Clary Sage": "클라리 세이지", "Orange Blossom": "오렌지 블로섬",
    "Bitter Orange": "비터 오렌지", "Pink Pepper": "핑크 페퍼", "Plum": "플럼",
    "Cherry": "체리", "Almond": "아몬드", "Coffee": "커피", "Cacao": "카카오",
}

_TERM_MAP_KO_OVERRIDE: dict[str, str] = {
    "향": "인센스",
    "가죽": "레더",
    "담배": "토바코",
    "생강": "진저",
}

PRODUCT_TYPE_MAP: dict[str, str] = {
    "eau de parfum": "Eau de Parfum",
    "edp": "Eau de Parfum",
    "eau de toilette": "Eau de Toilette",
    "edt": "Eau de Toilette",
    "parfum": "Parfum",
    "body spray": "Body Spray",
    "hair mist": "Hair Mist",
    "solid perfume": "Solid Perfume",
    "candle": "Candle",
}

CURRENCY_SYMBOL = {"USD": "$", "EUR": "€", "GBP": "£", "KRW": "₩"}

# 크롤링 품질 방어용 필터
# - 실제 향수/프래그런스 상품만 남기기
# - 404/컬렉션/메이크업/캔들/바디케어 제품이 섞이지 않도록 차단
ALLOWED_FRAGRANCE_TYPES = {"Eau de Parfum", "Eau de Toilette", "Parfum", "Body Spray", "Hair Mist", "Solid Perfume"}

NON_FRAGRANCE_TITLE_KEYWORDS = [
    "candle", "classic candle", "scented candle",
    "body oil", "shimmering body oil", "satin oil", "body lotion", "body cream",
    "moisturizer", "hand & body", "hand and body", "body wash", "shower gel",
    "deodorant", "beard oil", "soap",
    "eye", "lip", "powder", "primer", "quad", "foundation", "concealer", "mascara",
    "discovery set", "duo mini set", "set of", "gift set", "pre-composed",
]

BAD_TEXT_MARKERS = [
    "GLOSSARY ACCESSIBILITY", "NEED HELP", "TRACK MY ORDER", "SERVICES",
    "Benefits Set Includes", "Shades Included", "What Else You Need To Know",
    "Free From", "PROVEN RESULTS", "To discover labelling guidelines",
]

BAD_NOTE_PREFIXES = (
    "of ", "with ", "the ", "this ", "that ", "to ", "a ", "an ",
    "in ", "and ", "can be", "skin ", "benefits", "set includes",
)

BAD_NOTE_EXACT = {
    "in perfect harmony", "th", "s", "bl", "natural",
}


def apply_term_map(text: str) -> str:
    if not text:
        return text
    for en, ko in sorted(FRAGRANCE_TERM_MAP.items(), key=lambda x: len(x[0]), reverse=True):
        text = re.sub(
            rf"(?<![가-힣\w]){re.escape(en)}(?![가-힣\w])",
            ko,
            text,
            flags=re.IGNORECASE,
        )
    for wrong_ko, correct_ko in _TERM_MAP_KO_OVERRIDE.items():
        text = re.sub(rf"(?<![가-힣]){re.escape(wrong_ko)}(?![가-힣])", correct_ko, text)
    return text


def deepl_translate_text(text: str, target_lang: str = "KO", source_lang: str = "EN") -> str:
    """DeepL 단일 텍스트 번역. 키가 없거나 실패하면 원문에 용어 매핑만 적용해 반환."""
    if not text:
        return ""
    text = str(text).strip()
    if not text:
        return ""

    # 단일 향료명은 DeepL 호출 없이 매핑 우선
    lowered = text.lower()
    for en, ko in FRAGRANCE_TERM_MAP.items():
        if lowered == en.lower():
            return ko

    cache_key = (text, source_lang, target_lang)
    if cache_key in _translation_cache:
        return _translation_cache[cache_key]

    if not DEEPL_API_KEY:
        return apply_term_map(text)

    try:
        resp = requests.post(
            DEEPL_API_URL,
            headers={
                "Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={"text": [text], "source_lang": source_lang, "target_lang": target_lang},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        translated = data.get("translations", [{}])[0].get("text", "").strip()
        translated = apply_term_map(translated or text)
        _translation_cache[cache_key] = translated
        return translated
    except Exception as e:
        print(f"[DeepL 번역 실패] {text[:40]}... / {e}")
        return apply_term_map(text)


# ============================================================
# 데이터 모델
# ============================================================

@dataclass
class ProductNote:
    note_type: str
    note_name_original: str
    note_name_ko: str = ""
    sort_order: int = 0


@dataclass
class ProductImage:
    image_original_url: str
    sort_order: int = 0
    is_primary: bool = False
    image_internal_url: str = ""
    image_hash: str = ""
    download_status: str = "pending"


@dataclass
class Product:
    brand_name: str = ""
    country: str = ""
    source_site: str = ""
    source_url: str = ""
    source_product_id: str = ""

    product_name_original: str = ""
    product_name_ko: str = ""
    product_type: str = ""
    category: str = "Fragrance"
    scent_family: str = ""

    price_original: Optional[float] = None
    currency: str = "USD"
    volume_ml: Optional[float] = None
    stock_status: str = "unknown"

    description_original: str = ""
    description_ko: str = ""
    ingredients_original: str = ""
    ingredients_ko: str = ""
    allergen_list_json: list[str] = field(default_factory=list)

    images: list[ProductImage] = field(default_factory=list)
    notes: list[ProductNote] = field(default_factory=list)
    crawled_at: str = ""
    extra_attributes_json: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BrandConfig:
    key: str
    brand_name: str
    source_site: str
    country: str
    start_urls: list[str]
    allowed_domains: tuple[str, ...]
    default_currency: str


BRANDS: dict[str, BrandConfig] = {
    "tomford": BrandConfig(
        key="tomford",
        brand_name="Tom Ford",
        source_site="tomford_beauty",
        country="US",
        start_urls=["https://www.tomfordbeauty.com/collections/fragrance"],
        allowed_domains=("tomfordbeauty.com",),
        default_currency="USD",
    ),
    "diptyque": BrandConfig(
        key="diptyque",
        brand_name="Diptyque",
        source_site="diptyque_emea",
        country="BE",  # emea.diptyqueparis.com/en-be 기준. 필요 시 US/UK/EU로 변경
        start_urls=["https://emea.diptyqueparis.com/en-be/collections/all-fragrances"],
        allowed_domains=("diptyqueparis.com",),
        default_currency="EUR",
    ),
}


# ============================================================
# 공통 파싱 유틸
# ============================================================

def clean_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_price(text: str, default_currency: str = "USD") -> tuple[Optional[float], str]:
    if not text:
        return None, default_currency
    symbol_map = {"€": "EUR", "$": "USD", "£": "GBP", "₩": "KRW"}
    currency = default_currency
    for symbol, code in symbol_map.items():
        if symbol in text:
            currency = code
            break
    # 1,234.00 / 1 234,00 / 85,000원 모두 최대한 처리
    m = re.search(r"[\d][\d,\.\s]*", text)
    if not m:
        return None, currency
    raw = m.group(0).strip().replace(" ", "")
    if raw.count(",") == 1 and raw.count(".") == 0 and len(raw.split(",")[-1]) == 2:
        raw = raw.replace(",", ".")
    else:
        raw = raw.replace(",", "")
    try:
        return float(raw), currency
    except ValueError:
        return None, currency


def format_price(amount: Optional[float], currency: str) -> str:
    if amount is None:
        return ""
    if currency == "KRW":
        return f"{int(amount):,}원"
    symbol = CURRENCY_SYMBOL.get(currency, currency + " ")
    return f"{symbol}{amount:,.2f}"


def extract_volume(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"([\d.]+)\s*(ml|mL|ML)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*(fl\.?\s*)?oz", text, flags=re.IGNORECASE)
    if m:
        return round(float(m.group(1)) * 29.5735, 1)
    return None


def extract_product_type(text: str) -> str:
    """상품명/짧은 설명에서 제품 타입 추출.

    이전 버전은 페이지 전체 텍스트에서 타입을 찾다 보니,
    메이크업 상품도 네비게이션의 Body Spray 문구 때문에 Body Spray로 저장되는 문제가 있었습니다.
    이제는 호출부에서 상품명 중심으로 사용합니다.
    """
    lowered = (text or "").lower()
    for key, value in PRODUCT_TYPE_MAP.items():
        if key in lowered:
            return value
    return ""


def contains_bad_text(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker.lower() in lowered for marker in BAD_TEXT_MARKERS)


def clean_bad_page_text(text: str) -> str:
    """성분/노트 후보에 섞인 공통 UI 텍스트 제거."""
    if not text:
        return ""
    text = clean_text(text)
    if contains_bad_text(text):
        # 실제 성분이라면 보통 INCI처럼 긴 쉼표 목록이 나오는데,
        # 현재 문제 케이스는 공통 UI 문구뿐이므로 통째로 버립니다.
        return ""
    return text


def is_probably_fragrance_product(name: str, product_type: str, url: str = "") -> bool:
    title = (name or "").lower()
    if not title:
        return False
    if "404 not found" in title or "not found" in title:
        return False
    if any(kw in title for kw in NON_FRAGRANCE_TITLE_KEYWORDS):
        # 단, All Over Body Spray / Hair Mist는 향 제품으로 유지
        if "body spray" in title or "hair mist" in title:
            return True
        return False
    if product_type in ALLOWED_FRAGRANCE_TYPES:
        return True
    # 제품명에 향수 타입이 명시된 경우만 허용
    fragrance_tokens = ["eau de parfum", "eau de toilette", "parfum", "body spray", "hair mist", "solid perfume"]
    return any(tok in title for tok in fragrance_tokens)


def is_valid_note_text(text: str) -> bool:
    if not text:
        return False
    t = clean_text(text).strip(" .:-–—").strip()
    lower = t.lower()
    if not t or lower in BAD_NOTE_EXACT:
        return False
    if contains_bad_text(t):
        return False
    if len(t) < 2 or len(t) > 45:
        return False
    if len(t.split()) > 5:
        return False
    if lower.startswith(BAD_NOTE_PREFIXES):
        return False
    sentence_verbs = [" evokes ", " captures ", " reveals ", " features ", " contains ", " creates ", " leaves ", " helps ", " can be "]
    if any(v in f" {lower} " for v in sentence_verbs):
        return False
    # 설명문 조각으로 자주 섞인 표현 제거
    bad_fragments = ["impression", "senses", "formula", "finish", "skin on skin", "perfect harmony"]
    if any(b in lower for b in bad_fragments):
        return False
    return True


def make_product_id(url: str, name: str = "") -> str:
    base = url or name or str(time.time())
    return hashlib.md5(base.encode("utf-8")).hexdigest()[:12]


def parse_notes_text(text: str, note_type: str = "key") -> list[ProductNote]:
    if not text:
        return []
    text = clean_text(text)
    # 설명 문장 전체가 들어오는 것을 방지하기 위해 너무 긴 항목은 제외
    parts = re.split(r"[,/;•|]+|\band\b", text, flags=re.IGNORECASE)
    notes: list[ProductNote] = []
    for i, raw in enumerate(parts):
        name = raw.strip(" .:-–—\t\n")
        if not name or len(name) > 40:
            continue
        if re.search(r"\d", name):
            continue
        notes.append(ProductNote(note_type=note_type, note_name_original=name, sort_order=i))
    return notes


def parse_ingredients(text: str) -> tuple[str, list[str]]:
    text = clean_bad_page_text(text).strip(" .")
    if not text:
        return "", []
    allergens_keywords = [
        "limonene", "linalool", "geraniol", "citronellol", "eugenol", "coumarin",
        "cinnamal", "benzyl", "isoeugenol", "farnesol", "citral", "alpha-isomethyl ionone",
    ]
    found = []
    lower = text.lower()
    for kw in allergens_keywords:
        if kw in lower:
            found.append(kw)
    return text, found


def find_json_objects_in_script(raw: str) -> list[Any]:
    """script 안의 JSON-LD 또는 명확한 JSON 객체를 안전하게 일부 파싱."""
    results: list[Any] = []
    raw = raw.strip()
    if not raw:
        return results
    try:
        results.append(json.loads(raw))
        return results
    except Exception:
        pass
    # const productList = {...}; / window.__data = {...}; 류를 제한적으로 파싱
    for pattern in [
        r"const\s+productList\s*=\s*(\{.*?\});\s*\n",
        r"const\s+products\s*=\s*(\[.*?\]);\s*\n",
        r"window\.__INITIAL_STATE__\s*=\s*(\{.*?\});",
        r"window\.__PRELOADED_STATE__\s*=\s*(\{.*?\});",
        r"__NEXT_DATA__\s*=\s*(\{.*?\})\s*</script>",
    ]:
        m = re.search(pattern, raw, flags=re.DOTALL)
        if m:
            try:
                results.append(json.loads(m.group(1)))
            except Exception:
                continue
    return results


def walk_json(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_json(v)
    elif isinstance(obj, list):
        for item in obj:
            yield from walk_json(item)


def looks_like_product_url(href: str) -> bool:
    h = href.lower()
    bad = ["/cart", "/account", "/search", "/pages/", "/blogs/", "#", "javascript:"]
    if any(b in h for b in bad):
        return False
    # Tom Ford/Diptyque 모두 Shopify 계열이라 실제 상세는 /products/ 입니다.
    # 이전 버전은 /collections/eau-...까지 상품 후보로 잡아서 404가 저장됐습니다.
    return "/products/" in h


# ============================================================
# 상세 페이지 파서
# ============================================================

def parse_product_detail(response: Response, config: BrandConfig) -> Product:
    p = Product(
        brand_name=config.brand_name,
        country=config.country,
        source_site=config.source_site,
        source_url=str(response.url),
        source_product_id=make_product_id(str(response.url)),
        currency=config.default_currency,
        crawled_at=datetime.now(timezone.utc).isoformat(),
    )

    json_ld_products: list[dict[str, Any]] = []
    all_json_dicts: list[dict[str, Any]] = []

    for raw in response.css("script[type='application/ld+json']::text").getall():
        for parsed in find_json_objects_in_script(raw):
            for node in walk_json(parsed):
                all_json_dicts.append(node)
                ntype = node.get("@type")
                if ntype == "Product" or ntype == "ProductGroup" or (isinstance(ntype, list) and "Product" in ntype):
                    json_ld_products.append(node)

    product_json = json_ld_products[0] if json_ld_products else {}

    # 1) JSON-LD 우선
    p.product_name_original = clean_text(product_json.get("name", ""))
    p.description_original = clean_text(product_json.get("description", ""))

    brand = product_json.get("brand")
    if isinstance(brand, dict):
        p.extra_attributes_json["brand_name_from_page"] = brand.get("name", "")

    image = product_json.get("image")
    if isinstance(image, str):
        p.images.append(ProductImage(image_original_url=image, sort_order=0, is_primary=True))
    elif isinstance(image, list):
        for i, img in enumerate(image):
            if isinstance(img, str):
                p.images.append(ProductImage(image_original_url=img, sort_order=i, is_primary=(i == 0)))

    offers = product_json.get("offers")
    if isinstance(offers, list) and offers:
        offers = offers[0]
    if isinstance(offers, dict):
        price = offers.get("price") or offers.get("lowPrice")
        try:
            p.price_original = float(price) if price is not None else None
        except Exception:
            p.price_original, _ = extract_price(str(price), config.default_currency)
        p.currency = offers.get("priceCurrency") or p.currency
        avail = str(offers.get("availability", ""))
        if "InStock" in avail:
            p.stock_status = "in_stock"
        elif "OutOfStock" in avail:
            p.stock_status = "out_of_stock"

    # 2) 메타 태그 fallback
    if not p.product_name_original:
        p.product_name_original = clean_text(
            response.css("meta[property='og:title']::attr(content)").get()
            or response.css("h1::text").get()
            or ""
        )
    if not p.description_original:
        p.description_original = clean_text(
            response.css("meta[property='og:description']::attr(content)").get()
            or response.css("meta[name='description']::attr(content)").get()
            or ""
        )
    if not p.images:
        img = response.css("meta[property='og:image']::attr(content)").get()
        if img:
            p.images.append(ProductImage(image_original_url=urljoin(str(response.url), img), sort_order=0, is_primary=True))

    # 3) script 전체에서 상품성 데이터 보강
    scripts = response.css("script::text").getall()
    for raw in scripts:
        for parsed in find_json_objects_in_script(raw):
            for node in walk_json(parsed):
                if not isinstance(node, dict):
                    continue
                name = node.get("name") or node.get("title") or node.get("productName")
                if not p.product_name_original and isinstance(name, str) and 2 <= len(name) <= 120:
                    p.product_name_original = clean_text(name)
                if p.price_original is None:
                    price = node.get("price") or node.get("salePrice") or node.get("regularPrice")
                    if price:
                        if isinstance(price, dict):
                            amount = price.get("amount") or price.get("value")
                            curr = price.get("currency")
                        else:
                            amount, curr = price, None
                        try:
                            p.price_original = float(amount)
                        except Exception:
                            p.price_original, curr2 = extract_price(str(amount), config.default_currency)
                            curr = curr or curr2
                        if curr:
                            p.currency = str(curr)
                if not p.images:
                    img = node.get("image") or node.get("images")
                    if isinstance(img, str) and img.startswith("http"):
                        p.images.append(ProductImage(img, sort_order=0, is_primary=True))
                    elif isinstance(img, list):
                        for i, one in enumerate(img[:5]):
                            if isinstance(one, str) and one.startswith("http"):
                                p.images.append(ProductImage(one, sort_order=i, is_primary=(i == 0)))

    # 4) DOM 텍스트 기반 보강
    full_text = clean_text(" ".join(response.css("body *::text").getall()))

    if p.price_original is None:
        price_text = response.css("[class*='price']::text, [data-testid*='price']::text").get() or ""
        p.price_original, p.currency = extract_price(price_text or full_text[:2000], config.default_currency)

    # 제품 타입은 상품명 중심으로 판단합니다.
    # 페이지 전체 텍스트를 쓰면 네비게이션/추천 영역 때문에 메이크업도 Body Spray로 오분류될 수 있습니다.
    product_type_from_name = extract_product_type(p.product_name_original)
    product_type_from_desc = extract_product_type(p.description_original[:300])
    p.product_type = product_type_from_name or product_type_from_desc
    p.volume_ml = extract_volume(f"{p.product_name_original} {p.description_original[:500]}")

    # ingredients 후보
    ingr_match = re.search(
        r"(ingredients?|composition)[:\s]+(.{20,1200}?)(?:\s+(?:how to use|usage|delivery|shipping|returns|please note)|$)",
        full_text,
        flags=re.IGNORECASE,
    )
    if ingr_match:
        p.ingredients_original, p.allergen_list_json = parse_ingredients(ingr_match.group(2))

    # notes 후보: JSON-LD additionalProperty + DOM 텍스트
    for node in all_json_dicts:
        props = node.get("additionalProperty") or []
        if isinstance(props, dict):
            props = [props]
        for prop in props:
            if not isinstance(prop, dict):
                continue
            prop_name = str(prop.get("name", "")).lower()
            value = prop.get("value", "")
            if not value:
                continue
            if "top" in prop_name:
                p.notes.extend(parse_notes_text(str(value), "top"))
            elif "heart" in prop_name or "middle" in prop_name:
                p.notes.extend(parse_notes_text(str(value), "heart"))
            elif "base" in prop_name:
                p.notes.extend(parse_notes_text(str(value), "base"))
            elif "note" in prop_name or "ingredient" in prop_name:
                p.notes.extend(parse_notes_text(str(value), "key"))

    if not p.notes:
        note_patterns = [
            ("top", r"top notes?[:\s]+([^\.\n]{3,160})"),
            ("heart", r"(?:heart|middle) notes?[:\s]+([^\.\n]{3,160})"),
            ("base", r"base notes?[:\s]+([^\.\n]{3,160})"),
            ("key", r"(?:notes?|key notes?|olfactory notes?)[:\s]+([^\.\n]{3,160})"),
        ]
        for note_type, pattern in note_patterns:
            m = re.search(pattern, full_text, flags=re.IGNORECASE)
            if m:
                p.notes.extend(parse_notes_text(m.group(1), note_type))

    # 중복 노트 제거
    seen_notes: set[tuple[str, str]] = set()
    deduped: list[ProductNote] = []
    for n in p.notes:
        if not is_valid_note_text(n.note_name_original):
            continue
        key = (n.note_type, n.note_name_original.lower())
        if key not in seen_notes:
            seen_notes.add(key)
            n.sort_order = len(deduped)
            deduped.append(n)
    p.notes = deduped

    # 공통 UI 문구가 성분으로 들어간 경우 최종 방어
    p.ingredients_original = clean_bad_page_text(p.ingredients_original)

    return translate_product_fields(p)


def translate_product_fields(product: Product, sleep_sec: float = 0.1) -> Product:
    product.product_name_ko = deepl_translate_text(product.product_name_original)
    time.sleep(sleep_sec)
    product.description_ko = deepl_translate_text(product.description_original)
    time.sleep(sleep_sec)
    product.ingredients_ko = deepl_translate_text(product.ingredients_original)
    time.sleep(sleep_sec)
    for note in product.notes:
        note.note_name_ko = deepl_translate_text(note.note_name_original)
        time.sleep(0.03)
    return product


def product_to_mysql_row(prod: Product) -> dict[str, Any]:
    # top → heart → base → key 순서로 대표 향료 구성
    ordered_types = ["top", "heart", "base", "key"]
    key_ingredients: list[str] = []
    for t in ordered_types:
        vals = []
        for n in prod.notes:
            if n.note_type != t:
                continue
            raw = n.note_name_original
            ko = n.note_name_ko or apply_term_map(raw)
            if is_valid_note_text(raw) and is_valid_note_text(ko):
                vals.append(ko)
        if vals:
            key_ingredients = vals[:8]
            break

    return {
        "country": prod.country,
        "korean_name": prod.product_name_ko,
        "english_name": prod.product_name_original,
        "product_type": prod.product_type,
        "product_url": prod.source_url,
        "regular_price": format_price(prod.price_original, prod.currency),
        "image_url": prod.images[0].image_original_url if prod.images else "",
        "ingredients": prod.ingredients_ko,
        "key_ingredients": key_ingredients,
    }


# ============================================================
# Spider
# ============================================================

class MultiBrandFragranceSpider(Spider):
    name = "tomford_diptyque_fragrance"
    concurrency = 2
    download_delay = 2.0
    fetcher = "StealthyFetcher"

    def __init__(self, brand_keys: Optional[list[str]] = None, max_products_per_brand: int = 0):
        super().__init__()
        self.brand_keys = brand_keys or list(BRANDS.keys())
        self.max_products_per_brand = max_products_per_brand
        self.products: list[Product] = []
        self.raw_documents: list[dict[str, Any]] = []
        self.seen_urls: set[str] = set()
        self.brand_counts: dict[str, int] = {k: 0 for k in self.brand_keys}
        self.start_urls = []
        for key in self.brand_keys:
            self.start_urls.extend(BRANDS[key].start_urls)

    def _config_for_url(self, url: str) -> BrandConfig:
        host = urlparse(url).netloc.lower()
        for key in self.brand_keys:
            config = BRANDS[key]
            if any(domain in host for domain in config.allowed_domains):
                return config
        # fallback
        return BRANDS[self.brand_keys[0]]

    async def parse(self, response: Response):
        config = self._config_for_url(str(response.url))
        print(f"[목록] {config.brand_name} / {response.url}")

        product_urls = self._extract_product_urls(response, config)
        print(f"  후보 상품 URL: {len(product_urls)}개")

        for url in product_urls:
            if url in self.seen_urls:
                continue
            if self.max_products_per_brand and self.brand_counts[config.key] >= self.max_products_per_brand:
                continue
            self.seen_urls.add(url)
            self.brand_counts[config.key] += 1
            yield Request(url, callback=self.parse_detail)

        # pagination 후보
        for next_url in self._extract_next_urls(response, config):
            if next_url not in self.seen_urls:
                self.seen_urls.add(next_url)
                print(f"  [페이지네이션] {next_url}")
                yield Request(next_url, callback=self.parse)

    def _extract_product_urls(self, response: Response, config: BrandConfig) -> list[str]:
        base_url = str(response.url)
        urls: list[str] = []
        seen: set[str] = set()

        def add(raw_url: str):
            if not raw_url:
                return
            abs_url = urljoin(base_url, raw_url.split("?")[0])
            host = urlparse(abs_url).netloc.lower()
            if not any(domain in host for domain in config.allowed_domains):
                return
            if not looks_like_product_url(abs_url):
                return
            slug = urlparse(abs_url).path.lower().replace("-", " ").replace("/", " ")
            if any(kw in slug for kw in NON_FRAGRANCE_TITLE_KEYWORDS):
                if "body spray" not in slug and "hair mist" not in slug:
                    return
            if abs_url not in seen:
                seen.add(abs_url)
                urls.append(abs_url)

        # DOM 링크
        for href in response.css("a::attr(href)").getall():
            add(href)

        # script 내 url 필드
        for raw in response.css("script::text").getall():
            # 일반 URL 패턴
            for m in re.finditer(r'"(?:url|href|handle|path)"\s*:\s*"([^"]+)"', raw):
                add(m.group(1).replace("\\/", "/"))
            # JSON 파싱 가능한 경우
            for parsed in find_json_objects_in_script(raw):
                for node in walk_json(parsed):
                    if not isinstance(node, dict):
                        continue
                    for key in ["url", "href", "productUrl", "canonicalUrl", "path"]:
                        val = node.get(key)
                        if isinstance(val, str):
                            add(val)
        return urls

    def _extract_next_urls(self, response: Response, config: BrandConfig) -> list[str]:
        base_url = str(response.url)
        out: list[str] = []
        for href in response.css("a[rel='next']::attr(href), a[aria-label*='Next']::attr(href), a[href*='page=']::attr(href), a[href*='pageNumber=']::attr(href)").getall():
            next_url = urljoin(base_url, href)
            host = urlparse(next_url).netloc.lower()
            if any(domain in host for domain in config.allowed_domains):
                out.append(next_url)
        return list(dict.fromkeys(out))

    async def parse_detail(self, response: Response):
        config = self._config_for_url(str(response.url))
        try:
            product = parse_product_detail(response, config)
            if not product.product_name_original:
                print(f"  [스킵] 상품명 없음: {response.url}")
                return
            if not is_probably_fragrance_product(product.product_name_original, product.product_type, product.source_url):
                print(f"  [스킵] 향수 외/부정확 상품: {product.product_name_original} / type={product.product_type or '-'}")
                return

            self.products.append(product)
            row = product_to_mysql_row(product)
            self.raw_documents.append({
                "source_site": config.source_site,
                "brand_name": config.brand_name,
                "page_url": str(response.url),
                "page_type": "product_detail",
                "parsed_fields": row,
                "raw_summary": {
                    "description_original": product.description_original,
                    "ingredients_original": product.ingredients_original,
                    "notes_original": [asdict(n) for n in product.notes],
                    "images_count": len(product.images),
                },
                "crawled_at": datetime.now(timezone.utc).isoformat(),
            })

            print(
                f"  ✓ {config.brand_name} | {row['english_name']} | "
                f"{row['regular_price']} | notes={len(row['key_ingredients'])}"
            )
            yield row
        except Exception as e:
            print(f"  [상세 파싱 실패] {response.url} / {e}")

    async def on_close(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mysql_file = f"tomford_diptyque_mysql_ready_{timestamp}.json"
        raw_file = f"tomford_diptyque_nosql_raw_{timestamp}.json"

        rows = [product_to_mysql_row(p) for p in self.products]
        with open(mysql_file, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump(self.raw_documents, f, ensure_ascii=False, indent=2)

        print(f"\n[저장] MySQL ready → {mysql_file}")
        print(f"[저장] NoSQL raw   → {raw_file}")
        print(f"[완료] 총 {len(self.products)}개 상품")


# ============================================================
# 테스트 / 진입점
# ============================================================

def test_single_product(url: str):
    from scrapling.fetchers import StealthyFetcher, Fetcher

    print(f"[테스트] {url}")
    try:
        response = StealthyFetcher.fetch(url, headless=True, network_idle=True)
    except Exception as e:
        print(f"  StealthyFetcher 실패, 일반 Fetcher 시도: {e}")
        response = Fetcher.get(url)

    host = urlparse(url).netloc.lower()
    config = BRANDS["tomford"] if "tomford" in host else BRANDS["diptyque"]
    product = parse_product_detail(response, config)
    print(json.dumps(product_to_mysql_row(product), ensure_ascii=False, indent=2))
    print("\n[debug notes]")
    print(json.dumps([asdict(n) for n in product.notes], ensure_ascii=False, indent=2))


def main():
    args = [a.lower() for a in sys.argv[1:]]
    if args and args[0] == "test":
        if len(sys.argv) < 3:
            print("사용법: python tomford_diptyque_spider.py test [PRODUCT_URL]")
            return
        test_single_product(sys.argv[2])
        return

    if args and args[0] in BRANDS:
        brand_keys = [args[0]]
    else:
        brand_keys = list(BRANDS.keys())

    max_products = 0
    for a in args:
        if a.startswith("--max="):
            max_products = int(a.split("=", 1)[1])

    spider = MultiBrandFragranceSpider(brand_keys=brand_keys, max_products_per_brand=max_products)
    result = spider.start()
    print(f"[Spider result] completed={result.completed}, paused={result.paused}, items={len(result.items)}")


if __name__ == "__main__":
    main()
