import re

REDACTED = "[REDACTED]"

PATTERNS = [
    # AWS
    re.compile(r'(?:AKIA|ABIA|ACCA|ASIA)[0-9A-Z]{16}'),
    # Generic API key/token in assignments
    re.compile(
        r'''(?i)(?:api[_-]?key|api[_-]?secret|secret[_-]?key|access[_-]?token|auth[_-]?token|bearer)\s*[:=]\s*['"]?([A-Za-z0-9_\-/.+]{20,})['"]?'''
    ),
    # Bearer tokens in headers
    re.compile(r'(?i)Bearer\s+[A-Za-z0-9_\-/.+]{20,}'),
    # Private keys
    re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]*?-----END (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----'),
    # Password assignments
    re.compile(
        r'''(?i)(?:password|passwd|pwd)\s*[:=]\s*['"]?(\S{6,})['"]?'''
    ),
    # GitHub/GitLab tokens
    re.compile(r'(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}'),
    re.compile(r'glpat-[A-Za-z0-9_\-]{20,}'),
    # Slack tokens
    re.compile(r'xox[bpars]-[A-Za-z0-9\-]{10,}'),
    # Generic hex secrets (32+ chars, likely hashes/tokens)
    re.compile(r'(?i)(?:token|secret|key)\s*[:=]\s*["\']?([0-9a-f]{32,})["\']?'),
]


def redact(text: str) -> str:
    if not text:
        return text
    for pattern in PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text
