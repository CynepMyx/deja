from deja.parsers import claude_code, codex, review_extract

# Bump whenever a parser changes what text ends up in a chunk. The stored
# value is compared against this one on every `deja index`, and a mismatch
# forces a full rebuild — an index whose rows were produced by different
# parsing rules is not repairable by resuming from a byte offset.
#
# 2: assistant entries of one turn are accumulated instead of the turn ending
#    at the first of them, which used to drop the model's prose whenever a
#    turn opened with thinking or a tool call.
PARSER_VERSION = 2

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
