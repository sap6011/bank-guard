"""The API end to end, through FastAPI's test client.

Needs torch, the trained model and the demo bundle, so it is skipped in CI.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from fastapi.testclient import TestClient

from bankguard.api import BUNDLE, MODEL_DIR, app

pytestmark = pytest.mark.skipif(
    not (BUNDLE.exists() and (MODEL_DIR / "preprocessing.json").exists()),
    reason="needs the trained model and the demo bundle",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the startup once for all tests
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_list_filters_and_sorts(client):
    groups = client.app.state.table["group"]
    body = client.get("/transactions", params={"group": "hub", "limit": 200}).json()
    assert body["total"] == int(np.sum(groups == "hub"))
    assert all(item["group"] == "hub" and item["is_laundering"] for item in body["items"])
    scores = [item["score"] for item in body["items"]]
    assert scores == sorted(scores, reverse=True)


def test_detail_and_what_if_agree(client):
    top = client.get("/transactions", params={"sample": "laundering", "limit": 1}).json()["items"][0]
    detail = client.get(f"/transactions/{top['id']}").json()
    assert abs(detail["score"] - top["score"]) < 1e-5
    assert detail["past_transfers"], "the top laundering transfer should have a history"
    first = detail["past_transfers"][0]
    removed = {"removed": [{"side": first["side"], "slot": first["slot"]}]}
    after = client.post(f"/transactions/{top['id']}/what-if", json=removed).json()
    assert abs(after["score"] - (detail["score"] - first["effect"])) < 1e-5


def test_bad_requests_are_rejected(client):
    assert client.get("/transactions/999999").status_code == 404
    assert client.get("/transactions", params={"group": "nope"}).status_code == 422
    assert client.post("/transactions/0/what-if", json={"amount": -5}).status_code == 422


def test_model_card_matches_bundle(client):
    card = client.get("/model-card").json()
    groups = card["test_recall_by_group"]
    assert {g["group"] for g in groups} == {"patterned", "quiet", "hub"}
    assert sum(g["laundering"] for g in groups) == card["test_laundering"]
    thresholds = [a["threshold"] for a in card["alert_thresholds"]]
    assert np.allclose(thresholds, client.app.state.scorer.data["alert_threshold"])
    assert len(card["limitations"]) >= 4

def test_time_sort_is_chronological(client):
    items = client.get("/transactions", params={"sort": "time", "limit": 200}).json()["items"]
    times = [item["transfer"]["time"] for item in items]
    assert times == sorted(times)