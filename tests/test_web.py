"""The FastAPI service, against a synthetic study so no download is needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from redrawn import web
from redrawn.data import Study

N_TRACTS = 40


@pytest.fixture
def client(monkeypatch):
    rng = np.random.default_rng(0)
    n = rng.integers(50, 500, N_TRACTS)
    tracts = pd.DataFrame(
        {
            "tract": np.arange(N_TRACTS),
            "geoid": [f"g{i}" for i in range(N_TRACTS)],
            "borough": ["Queens"] * N_TRACTS,
            "nta2020": [f"n{i // 4}" for i in range(N_TRACTS)],
            "ntaname": [f"Area {i // 4}" for i in range(N_TRACTS)],
            "cdta2020": [f"c{i // 8}" for i in range(N_TRACTS)],
            "n": n,
            "injury": rng.integers(0, 40, N_TRACTS),
            "distraction": rng.integers(0, 30, N_TRACTS),
            "speeding": rng.integers(0, 12, N_TRACTS),
            "vru": rng.integers(0, 10, N_TRACTS),
            "unspecified": rng.integers(0, 25, N_TRACTS),
            "lon": rng.random(N_TRACTS) - 74,
            "lat": rng.random(N_TRACTS) + 40.5,
            "area": rng.random(N_TRACTS),
            "keep": True,
        }
    )
    study = Study(
        crashes=pd.DataFrame({"injury": [0, 1], "speeding": [0, 1], "year": [2019, 2020]}),
        tracts=tracts,
        zones=pd.DataFrame({"layer": [], "zone": [], "n": []}),
        report={"years": [2013, 2025]},
    )
    monkeypatch.setattr(web, "load_study", lambda: study)
    return TestClient(web.build_app())


def test_summary_endpoint(client):
    got = client.get("/api/summary").json()
    assert got["tracts"] == N_TRACTS
    assert got["crashes"] > 0


def test_score_endpoint_matches_the_library(client):
    labels = [i // 4 for i in range(N_TRACTS)]
    response = client.post("/api/score", json={"labels": labels, "pair": "speeding"})
    assert response.status_code == 200
    got = response.json()
    assert got["k"] == 10
    assert -1.0 <= got["r"] <= 1.0


def test_score_endpoint_rejects_the_wrong_length(client):
    response = client.post("/api/score", json={"labels": [0, 1, 2]})
    assert response.status_code == 400
    assert "length" in response.json()["detail"]


def test_score_endpoint_rejects_an_unknown_pair(client):
    labels = [i // 4 for i in range(N_TRACTS)]
    response = client.post("/api/score", json={"labels": labels, "pair": "weather"})
    assert response.status_code == 400


def test_score_endpoint_rejects_negative_labels(client):
    labels = [-1] * N_TRACTS
    response = client.post("/api/score", json={"labels": labels})
    assert response.status_code == 400


def test_weighted_flag_changes_the_answer(client):
    labels = [i // 4 for i in range(N_TRACTS)]
    plain = client.post("/api/score", json={"labels": labels}).json()["r"]
    weighted = client.post(
        "/api/score", json={"labels": labels, "weighted": True}
    ).json()["r"]
    assert plain != pytest.approx(weighted)


def test_index_is_served(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "redrawn" in response.text


def test_the_page_is_self_contained():
    """No CDN, no external stylesheet, no remote font: the page must work with
    nothing but the files in this repository."""
    from redrawn.config import UI

    import re

    page = (UI / "index.html").read_text()
    # Anything that would actually fetch from another host. The bare string
    # "http://" is not enough: the SVG namespace URI contains it and is never
    # dereferenced.
    for pattern in (
        r'src\s*=\s*["\']https?://',
        r'href\s*=\s*["\']https?://',
        r"@import",
        r"//cdn\.",
        r"fonts\.googleapis",
    ):
        assert not re.search(pattern, page), f"external reference found: {pattern}"
    # The one script it loads is the local analysis port.
    assert page.count("<script src=") == 1
    assert 'src="maup.js"' in page
