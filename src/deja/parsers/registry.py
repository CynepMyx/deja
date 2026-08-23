from deja.parsers import claude_code, codex, review_extract

PARSERS = {
    claude_code.SOURCE: claude_code,
    codex.SOURCE: codex,
    review_extract.SOURCE: review_extract,
}


def get_parser(source: str):
    if source not in PARSERS:
        raise ValueError(
            f"Unknown source '{source}'. Available: {sorted(PARSERS)}"
        )
    return PARSERS[source]


def all_sources() -> list[str]:
    return sorted(PARSERS)
