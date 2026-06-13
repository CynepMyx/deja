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
    "dop_v1_151a47ab23c4def567890abcdef1234567890abcd",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTYifQ.SflKxwRJSMeKKF2QT4fwpM",
    "AIzaSyA-1234567890abcdefghijklmnopqrstuv",
    "sk_live_4eC39HqLyjWDarjtT1zdp7dc12345",
    "1234567890:AAEhBOweik6ad9r_QXMENQjcrGbqCr4K-pc",
    "npm_AbCd1234567890efGhIjKlMnOpQrStUvWx",
])
def test_redact_bare_tokens(token):
    text = f"env output:\n{token}\nnext line"
    result = redact(text)
    assert token not in result, f"Bare token leaked: {token[:12]}..."
    assert REDACTED in result


def test_redact_still_preserves_normal_text():
    text = "Fix the skirt-and-blouse layout for nginx proxy in project alpha"
    assert redact(text) == text
