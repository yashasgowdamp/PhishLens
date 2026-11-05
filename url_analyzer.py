import ipaddress
import re
from urllib.parse import urlsplit

from risk_engine import assess_url_risk

MAX_URL_LENGTH = 2_048
SHORTENER_HOSTS = {
    "bit.ly",
    "goo.gl",
    "is.gd",
    "ow.ly",
    "rebrand.ly",
    "shorturl.at",
    "t.co",
    "tinyurl.com",
    "t.ly",
}
SUSPICIOUS_WORDS = re.compile(
    r"(?:^|[./_-])(account|auth|billing|login|password|pay|secure|signin|"
    r"update|verify|wallet)(?:$|[./_-])",
    re.IGNORECASE,
)


class URLAnalysisError(ValueError):
    """Raised when a submitted URL is malformed or outside the supported scope."""


def analyze_url(raw_url):
    if not isinstance(raw_url, str) or not raw_url.strip():
        raise URLAnalysisError("Please provide a URL.")
    if len(raw_url) > MAX_URL_LENGTH:
        raise URLAnalysisError("URL exceeds the 2,048 character limit.")
    if any(ord(character) < 32 or character.isspace() for character in raw_url):
        raise URLAnalysisError("URL must not contain whitespace or control characters.")

    value = raw_url.strip()
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError as error:
        raise URLAnalysisError("Please provide a valid http:// or https:// URL.") from error

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc or not hostname:
        raise URLAnalysisError("Please provide a valid http:// or https:// URL.")
    if parsed.username is not None or parsed.password is not None:
        raise URLAnalysisError("URLs containing embedded credentials are not accepted.")
    if port is not None and not 1 <= port <= 65535:
        raise URLAnalysisError("URL port must be between 1 and 65535.")

    hostname = hostname.lower().rstrip(".")
    indicators = []

    if len(value) > 100:
        indicators.append("Unusually long URL")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        indicators.append("URL contains an IP address instead of a domain")

    if "@" in value:
        indicators.append("URL contains an @ symbol")
    if re.search(r"%(?:00|0[0-9a-f]|1[0-9a-f]|2f|5c)|\\", value, re.IGNORECASE):
        indicators.append("URL contains suspicious encoded or escaped characters")

    if hostname in SHORTENER_HOSTS:
        indicators.append("URL uses a known shortening service")
    if any(label.startswith("xn--") for label in hostname.split(".")):
        indicators.append("URL hostname contains punycode")
    if len(hostname.split(".")) >= 5:
        indicators.append("URL hostname has many subdomains")
    if hostname.count("-") >= 3:
        indicators.append("URL hostname contains several hyphens")
    if parsed.scheme.lower() == "http":
        indicators.append("URL uses an insecure HTTP scheme")
    if SUSPICIOUS_WORDS.search(f"{hostname}{parsed.path}"):
        indicators.append("URL contains account or credential-related wording")
    if parsed.path.count("/") >= 5:
        indicators.append("URL contains an unusual number of path separators")

    risk = assess_url_risk(indicators)
    if risk["risk_level"] == "high":
        classification = "Suspicious"
    elif risk["risk_level"] == "medium":
        classification = "Review required"
    else:
        classification = "No suspicious indicators detected"

    return {
        "url": value,
        "hostname": hostname,
        "indicators": indicators,
        "classification": classification,
        "risk": risk,
        "risk_level": risk["risk_level"],
        "risk_score": risk["risk_score"],
        "warning_reasons": risk["warning_reasons"],
    }
