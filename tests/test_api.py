"""
API tests — no model downloads.

We override only the detector dependency with a fake whose verdict flips when
trust-comments are present (mimicking a model fooled by camouflage). The real
Attacker and real defense pipeline run as-is, since they work offline. This lets
us assert the headline behavior: the attack flips the verdict, the defense
restores it.
"""

from fastapi.testclient import TestClient

from src.api import app, get_detector
from src.detector import Prediction


class FakeDetector:
    """Predicts 'vulnerable' UNLESS the code contains fake-trust banners — i.e.
    it gets fooled by the camouflage, exactly like the real model does."""
    model_name = "fake/detector"
    is_loaded = True

    def predict(self, code: str) -> Prediction:
        fooled = any(k in code for k in ("CodeQL", "CERT", "VERIFIED_SAFE"))
        if fooled:
            return Prediction(False, "not vulnerable", 0.95, 0.05)
        return Prediction(True, "vulnerable", 0.95, 0.95)


# Wire the fake into the app for every test in this file.
app.dependency_overrides[get_detector] = lambda: FakeDetector()
client = TestClient(app)

VULN_CODE = "char buf[64];\nstrcpy(buf, user_input);\nreturn buf;"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.json()["detector_model"] == "fake/detector"


def test_predict_flags_clean_vulnerable_code():
    r = client.post("/predict", json={"code": VULN_CODE})
    assert r.status_code == 200
    assert r.json()["is_vulnerable"] is True


def test_attack_flips_the_verdict():
    r = client.post("/attack", json={"code": VULN_CODE})
    body = r.json()
    assert body["before"]["is_vulnerable"] is True      # caught when clean
    assert body["after"]["is_vulnerable"] is False       # fooled after camouflage
    assert body["flipped"] is True
    assert "CodeQL" in body["injected_code"]


def test_pipeline_tells_the_full_story():
    r = client.post("/pipeline", json={"code": VULN_CODE})
    body = r.json()
    assert body["clean"]["prediction"]["is_vulnerable"] is True
    assert body["injected"]["prediction"]["is_vulnerable"] is False
    assert body["defended"]["prediction"]["is_vulnerable"] is True   # restored!
    assert body["attack_flipped"] is True
    assert body["defense_restored"] is True
    assert body["defended"]["prediction"]["is_vulnerable"] == \
        body["clean"]["prediction"]["is_vulnerable"]


def test_invalid_request_is_rejected():
    r = client.post("/predict", json={})        # missing required 'code'
    assert r.status_code == 422                  # pydantic validation error


def teardown_module():
    app.dependency_overrides.clear()
