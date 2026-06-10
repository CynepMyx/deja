from deja.parsers import claude_code

PARSERS = {
    claude_code.SOURCE: claude_code,
}


def get_parser(source: str):
    if source not in PARSERS:
        raise ValueError(
            f"Unknown source '{source}'. Available: {sorted(PARSERS)}"
        )
    return PARSERS[source]


def all_sources() -> list[str]:
    return sorted(PARSERS)
