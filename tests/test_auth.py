import pytest


@pytest.mark.parametrize(
    "method,path", [("GET", "/ready"), ("GET", "/version"), ("POST", "/embed"), ("POST", "/chunks")]
)
@pytest.mark.parametrize(
    "token", [None, "Bearer wrong", "Basic abc", "Bearer", "Bearer " + "x" * 32]
)
def test_auth_first(client, backend, method, path, token):
    headers = {} if token is None else {"Authorization": token}
    response = client.request(method, path, headers=headers, content=b"not json")
    assert response.status_code == 401
    assert response.json() == {"error": "unauthorized"}
    assert backend.calls == []
