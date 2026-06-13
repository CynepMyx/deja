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
    # Password assignments (EN + RU)
    re.compile(
        r'''(?i)(?:password|passwd|pwd|пароль|Password)\s*[:=]\s*['"`]?(\S{6,})['"`]?'''
    ),
    # -p flag (mysql -pSECRET)
    re.compile(r'(?<=-p)[A-Za-z0-9!@#$%^&*_\-]{8,}(?=[\s\'"\)]|$)'),
    # GitHub/GitLab tokens
    re.compile(r'(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36,}'),
    re.compile(r'glpat-[A-Za-z0-9_\-]{20,}'),
    # Slack tokens
    re.compile(r'xox[bpars]-[A-Za-z0-9\-]{10,}'),
    # Cloudflare/Supabase/Grafana service tokens
    re.compile(r'(?:cfut|sbp|glsa)_[A-Za-z0-9_\-]{20,}'),
    # Generic hex secrets (32+ chars, likely hashes/tokens)
    re.compile(r'(?i)(?:token|secret|key)\s*[:=]\s*["\']?([0-9a-f]{32,})["\']?'),
    # Russian context: "пароль `X`" or "пароль: X" or "пароль X"
    re.compile(r'пароль\s*[:]\s*[`\'"]?([A-Za-z0-9!@#$%^&*()_\-+=,.]{6,})[`\'"]?'),
    re.compile(r'пароль\s+`([^`]{6,})`'),
    # sshpass -p 'SECRET'
    re.compile(r'''sshpass\s+-p\s*['"]?([^\s'"]{6,})['"]?'''),
    # URL credentials: proto://user:pass@host
    re.compile(r'://[^:/@\s]+:([^@\s]{6,})@'),
    # login / password pairs (slash separated)
    re.compile(r'(?i)(?:логин|login)\s*/\s*\S+\s*/\s*(\S{6,})'),
    # Truncation fallback: BEGIN header without END still gets redacted
    re.compile(r'-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----[\s\S]+'),
    # Bare well-known token formats (appear in env dumps / configs without assignment context)
    re.compile(r'\bsk-(?:ant|proj)-[A-Za-z0-9_\-]{16,}'),
    re.compile(r'\b(?:dop_v1_|doo_v1_|dor_v1_|figd_)[A-Za-z0-9_\-]{15,}'),
    re.compile(r'\bnpm_[A-Za-z0-9]{36}\b'),
    re.compile(r'\bpypi-AgEIc[A-Za-z0-9_\-]{20,}'),
    re.compile(r'\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}'),  # JWT
    re.compile(r'\bAIza[0-9A-Za-z_\-]{35}'),
    re.compile(r'\b[rs]k_live_[A-Za-z0-9]{20,}'),  # Stripe
    re.compile(r'\b\d{7,}:AA[A-Za-z0-9_\-]{33,}'),  # Telegram bot
]


def redact(text: str) -> str:
    if not text:
        return text
    for pattern in PATTERNS:
        text = pattern.sub(REDACTED, text)
    return text


def redact_turn(turn: dict) -> dict:
    turn["user_text"] = redact(turn["user_text"])
    turn["assistant_text"] = redact(turn["assistant_text"])
    if turn.get("tool_result_text"):
        turn["tool_result_text"] = redact(turn["tool_result_text"])
    return turn
