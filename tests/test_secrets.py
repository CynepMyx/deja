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
