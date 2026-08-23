import pytest

from deja.secrets import redact, REDACTED


def test_redact_aws_key():
    text = "key = AKIAIOSFODNN7EXAMPLE"
    result = redact(text)
    assert "AKIAIOSFODNN7EXAMPLE" not in result
    assert REDACTED in result


def test_redact_bearer_token():
    text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.abc123"
    result = redact(text)
    assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result


def test_redact_password_assignment():
    text = "password=SuperSecret123!"
    result = redact(text)
    assert "SuperSecret123!" not in result


def test_redact_github_token():
    text = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
    result = redact(text)
    assert "ghp_" not in result


def test_redact_preserves_normal_text():
    text = "Fix the SSL certificate for nginx proxy server"
    result = redact(text)
    assert result == text


def test_redact_api_key_in_config():
    text = "api_key: 'sk-proj-abcdef1234567890abcdef1234567890'"
    result = redact(text)
    assert "sk-proj" not in result


def test_redact_empty():
    assert redact("") == ""
    assert redact(None) is None


def test_redact_private_key():
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAKCAQEA\n-----END RSA PRIVATE KEY-----"
    result = redact(text)
    assert "MIIEpAIBAAKCAQEA" not in result


def test_truncated_private_key_redacted():
    text = "-----BEGIN OPENSSH PRIVATE KEY-----\n" + "A" * 3000  # END cut off
    result = redact(text)
    assert "AAAA" not in result
    assert REDACTED in result


@pytest.mark.parametrize("token", [
    "sk-ant-api03-AbCdEf123456789012345678901234",
    "sk-proj-AbCdEf12345678901234567890",
    "dop_v1_0f1e2d3c4b5a69780f1e2d3c4b5a69780f1e2d3c4",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.SflKxwRJSMeKKF2QT4fwpM",
    "AIzaSyA-1234567890abcdefghijklmnopqrstuv",
    "sk_live_4eC39HqLyjWDarjtT1zdp7dc12345",
    "1234567890:AAEhBOweik6ad9r_QXMENQjcrGbqCr4K-pc",
    "npm_AbCd1234567890efGhIjKlMnOpQrStUvWxYz",
])
def test_redact_bare_tokens(token):
    text = f"env output:\n{token}\nnext line"
    result = redact(text)
    assert token not in result, f"Bare token leaked: {token[:12]}..."
    assert REDACTED in result


def test_redact_still_preserves_normal_text():
    text = "Fix the skirt-and-blouse layout for nginx proxy in project alpha"
    assert redact(text) == text


@pytest.mark.parametrize("token", [
    "12345678901:AAEhBOweik6ad9r_QXMENQjcrGbqCr4K-pc",  # 11-digit Telegram ID
    "1234567:AAEhBOweik6ad9r_QXMENQjcrGbqCr4K-pc",      # 7-digit legacy Telegram ID
    "pypi-AgEIcHRlc3QucHlwaS5vcmc" + "x" * 30,           # TestPyPI
    "doo_v1_" + "a1b2c3d4e5" * 4,                        # DO OAuth
    "dor_v1_" + "a1b2c3d4e5" * 4,                        # DO refresh
    "npm_" + "Ab12Cd34Ef56Gh78Ij90Kl12Mn34Op56Qr78",     # real npm token shape (36 base62)
])
def test_redact_bare_tokens_variants(token):
    result = redact(f"output:\n{token}\nend")
    assert token not in result
    assert REDACTED in result


def test_npm_env_vars_not_redacted():
    text = "npm_package_version=1.2.3 npm_config_registry=https://r.npmjs.org npm_lifecycle_event=postinstall"
    assert redact(text) == text
