import pytest


@pytest.fixture(autouse=True)
def _runtime_env(monkeypatch):
    monkeypatch.setenv("GVA_HASH_PEPPER", "pepper-" + ("p" * 32))
    monkeypatch.setenv("XAI_API_KEY", "xai-" + ("k" * 20))
    monkeypatch.setenv("GVA_API_TOKEN_SECRET", "token-" + ("t" * 24))
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_KEY", raising=False)
    monkeypatch.delenv("INVOCATION_ID", raising=False)
