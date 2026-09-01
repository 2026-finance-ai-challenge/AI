from __future__ import annotations

import re

US_STATE_CODE_PATTERN = (
    r"AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|IA|ID|IL|IN|KS|KY|LA|MA|MD|ME|MI|MN|MO|MS|"
    r"MT|NC|ND|NE|NH|NJ|NM|NV|NY|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VA|VT|WA|WI|WV|WY"
)


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_whitespace(value).strip(" ,;:")
    normalized = re.sub(r"(?<=\b[A-Z])\.(?=[A-Z])", ". ", normalized)
    normalized = re.sub(r"[^A-Za-z0-9 .,'-]", " ", normalized)
    normalized = normalize_whitespace(normalized)
    return normalized or None


def canonicalize_name(value: str | None) -> str:
    return "" if not value else re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_address(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[\n\r\t,]+", " ", value)
    normalized = re.sub(r"[^A-Za-z0-9#.' -]", " ", normalized)
    normalized = normalize_whitespace(normalized)
    normalized = _normalize_zip_like_tokens(normalized)
    normalized = _normalize_common_address_ocr_noise(normalized)
    normalized = _canonicalize_address_country_tail(normalized)
    return normalized or None


def _normalize_zip_like_tokens(value: str) -> str:
    digit_like_map = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "I": "1",
            "l": "1",
            "L": "1",
            "S": "5",
            "B": "8",
        }
    )

    def _replace(match: re.Match[str]) -> str:
        token = match.group(0)
        if len(token) != 5 or sum(character.isdigit() for character in token) < 3:
            return token
        normalized_token = token.translate(digit_like_map)
        if normalized_token.isdigit():
            return normalized_token
        return token

    return re.sub(r"\b[A-Za-z0-9]{5}\b", _replace, value)


def _normalize_common_address_ocr_noise(value: str) -> str:
    normalized = re.sub(r"\bAp[l1I]\b(?=\s+[0-9A-Za-z])", "Apt", value, flags=re.IGNORECASE)
    normalized = normalized.replace(".", " ")
    normalized = re.sub(
        r"\b(Apt|Suite|Ste|Unit)(?=\d)",
        r"\1 ",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\bB[i1l]vd\b", "Blvd", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\bBvd\b", "Blvd", normalized, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\b(Blvd|Boulevard|Street|St|Road|Rd|Avenue|Ave)(?=(?:Apt|Suite|Ste|Unit)\b)",
        r"\1 ",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\b(\d{5})(?=U(?:n|r)ited\b)", r"\1 ", normalized)
    normalized = re.sub(r"\bUrited\b", "United", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\b(\d+[A-Za-z])(?=[A-Z][a-z]{2,}\b)", r"\1 ", normalized)
    normalized = re.sub(
        rf"\b([A-Za-z]{{3,}})(?=(?:{US_STATE_CODE_PATTERN})\b)",
        r"\1 ",
        normalized,
    )
    normalized = re.sub(
        r"\bLos\s+A(?:r|n)(?:i|1|l)?geles\b",
        "Los Angeles",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\bLos\s+Ange[i1l]es\b",
        "Los Angeles",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"\bAm[o0]r[i1]ca\b", "America", normalized, flags=re.IGNORECASE)
    return normalize_whitespace(normalized)


def _canonicalize_address_country_tail(value: str) -> str:
    country_match = re.search(r"\b(?:United\s+States|USA)\b", value, re.IGNORECASE)
    if not country_match:
        return value
    prefix = value[: country_match.start()].rstrip()
    normalized_country = normalize_country(value[country_match.start() :])
    if normalized_country != "United States of America":
        return value
    return " ".join(part for part in [prefix, normalized_country] if part).strip()


def normalize_country(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_whitespace(value)
    compact_alpha = re.sub(r"[^A-Za-z0-9]", "", normalized).lower()
    compact_alpha = compact_alpha.translate(str.maketrans({"0": "o", "1": "i", "5": "s"}))
    if re.search(r"united states(?: of america)?", normalized, re.IGNORECASE):
        return "United States of America"
    if "nitedstates" in compact_alpha and ("america" in compact_alpha or "amerca" in compact_alpha):
        return "United States of America"
    if re.search(r"\busa\b", normalized, re.IGNORECASE):
        return "United States of America"
    return normalized or None


def normalize_country_code(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = normalize_country(value)
    if normalized == "United States of America":
        return "US"
    compact = normalize_whitespace(value).upper()
    compact = compact.translate(str.maketrans({"0": "O", "1": "I", "5": "S"}))
    compact = re.sub(r"[^A-Z]", "", compact)
    if compact == "US":
        return "US"
    match = re.search(r"\b([A-Z]{2})\b", value.upper())
    return match.group(1) if match else None


def normalize_percentage(value: str | None) -> str | None:
    if value is None:
        return None
    match = re.search(r"(\d{1,2})", value)
    if not match:
        return None
    return f"{match.group(1)}%"


def normalize_english_date(value: str | None) -> str | None:
    if not value:
        return None
    normalized = normalize_whitespace(value).strip(" ,.;:")
    normalized = re.sub(r"^Date\s*[:;]?\s*", "", normalized, flags=re.IGNORECASE)
    match = re.search(
        r"([A-Za-z]+)[\s.,]+(?:1\s+)?(\d{1,2})(?:st|nd|rd|th)?[,.]?\s*(\d{4})",
        normalized,
        re.IGNORECASE,
    )
    if not match:
        return normalized or None
    month, day, year = match.groups()
    return f"{month} {int(day)}, {year}"


def normalize_iso_date(value: str | None) -> str | None:
    if not value:
        return None
    normalized = re.sub(r"\s+", "", value)
    match = re.search(r"(\d{4})[년./-]?(\d{1,2})[월./-]?(\d{1,2})", normalized)
    if not match:
        return normalize_whitespace(value)
    year, month, day = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}"


def normalize_apostille_date(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = normalize_whitespace(value)
    cleaned = re.sub(r"_+", " ", cleaned)
    cleaned = re.sub(r"\b(DAY)\s*(OF)\s*([A-Za-z]+)", r"\1 \2 \3", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"([A-Za-z])(\d{4})", r"\1, \2", cleaned)
    cleaned = re.sub(r",\s*(\d{4})", r", \1", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,;:")
    if not re.search(r"\b\d{4}\b", cleaned):
        return None
    return cleaned or None


def normalize_apostille_authority(value: str | None) -> str | None:
    if not value:
        return None
    cleaned = value.replace("StateState", "State State")
    cleaned = cleaned.replace("Deputy Secretary of StateState", "Deputy Secretary of State State")
    cleaned = re.sub(r"(?<=[A-Za-z]),(?=[A-Za-z])", ", ", cleaned)
    return normalize_whitespace(cleaned) or None
