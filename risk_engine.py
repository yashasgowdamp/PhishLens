RISK_THRESHOLDS = {
    "low": (0, 30),
    "medium": (31, 69),
    "high": (70, 100),
}

_URL_INDICATOR_WEIGHTS = {
    "Unusually long URL": 10,
    "URL contains an IP address instead of a domain": 25,
    "URL contains an @ symbol": 18,
    "URL contains suspicious encoded or escaped characters": 12,
    "URL uses a known shortening service": 18,
    "URL hostname contains punycode": 15,
    "URL hostname has many subdomains": 12,
    "URL hostname contains several hyphens": 8,
    "URL uses an insecure HTTP scheme": 10,
    "URL contains account or credential-related wording": 12,
    "URL contains an unusual number of path separators": 8,
}


def assess_risk(base_score, warning_reasons):
    """Return a bounded, explainable risk score; it is not an attack probability."""
    score = min(100, max(0, int(base_score)))
    if score <= RISK_THRESHOLDS["low"][1]:
        level = "low"
    elif score <= RISK_THRESHOLDS["medium"][1]:
        level = "medium"
    else:
        level = "high"
    return {
        "risk_score": score,
        "risk_level": level,
        "warning_reasons": list(dict.fromkeys(warning_reasons)),
        "is_probability": False,
    }


def assess_url_risk(indicators):
    reasons = list(indicators)
    score = min(
        100,
        sum(_URL_INDICATOR_WEIGHTS.get(indicator, 0) for indicator in reasons),
    )
    if not reasons:
        reasons = ["No configured suspicious URL indicators were detected."]
    return assess_risk(score, reasons)


def combine_email_risk(classification, confidence, url_result=None):
    normalized_classification = classification.strip().lower()
    if normalized_classification == "spam":
        score = 55
        reasons = ["The trained email model classified this message as Spam."]
        if confidence >= 0.8:
            score += 15
            reasons.append("The model's predicted class confidence is at least 0.80.")
    elif normalized_classification == "legitimate":
        score = 5
        reasons = ["The trained email model classified this message as Legitimate."]
        if confidence < 0.6:
            score += 20
            reasons.append("The model has low confidence in its predicted class.")
    else:
        raise ValueError("classification must be Spam or Legitimate")

    if url_result:
        score += url_result["risk"]["risk_score"]
        reasons.extend(url_result["risk"]["warning_reasons"])

    return assess_risk(score, reasons)
