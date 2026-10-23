import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as webapp  # noqa: E402
import detector  # noqa: E402
from detector import Detection  # noqa: E402


class FakeDetector:
    """Stands in for the YOLO model: 'finds' a weapon in any bright frame."""

    def detect(self, frame):
        return [Detection("weapon", 0.91, (10, 10, 40, 30))] if frame.mean() > 100 else []


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "RESULT_DIR", tmp_path)
    monkeypatch.setattr(webapp, "_detector", FakeDetector())
    monkeypatch.setattr(webapp, "_detector_error", None)
    webapp.app.config["TESTING"] = True
    return webapp.app.test_client()


def png_bytes(value):
    ok, buf = cv2.imencode(".png", np.full((120, 160, 3), value, np.uint8))
    return io.BytesIO(buf.tobytes())


def test_image_with_weapon(client):
    r = client.post("/detect", data={"file": (png_bytes(255), "photo.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200 and b"Weapon detected" in r.data and b"weapon 91%" in r.data


def test_image_without_weapon(client):
    r = client.post("/detect", data={"file": (png_bytes(0), "photo.png")},
                    content_type="multipart/form-data")
    assert b"No weapon detected" in r.data


def test_path_traversal_name_is_not_used_on_disk(client):
    def listing():  # project root, ignoring caches pytest itself creates
        return {p.name for p in webapp.BASE_DIR.iterdir() if "cache" not in p.name}

    before = listing()
    r = client.post("/detect", data={"file": (png_bytes(255), "../../evil.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200
    assert not (webapp.BASE_DIR / "evil.png").exists()
    assert listing() == before
    assert [p.name for p in webapp.UPLOAD_DIR.iterdir() if p.name != ".gitkeep"] == []  # upload deleted


def test_rejects_unsupported_type(client):
    r = client.post("/detect", data={"file": (io.BytesIO(b"x"), "script.py")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert b"Unsupported file type" in r.data


def test_video_scan(client, tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(30):
        writer.write(np.full((120, 160, 3), 255 if i >= 20 else 0, np.uint8))
    writer.release()
    hits, checked, fps = detector.scan_video(FakeDetector(), str(path), every_n_frames=5)
    assert checked == 6 and len(hits) == 2 and hits[0][0] == pytest.approx(2.0)


def test_missing_weights_message(monkeypatch, tmp_path):
    monkeypatch.setenv("WEAPON_WEIGHTS", str(tmp_path / "nope.weights"))
    with pytest.raises(FileNotFoundError, match="weights not found"):
        detector.WeaponDetector()


def test_config_parses_with_one_class():
    net = cv2.dnn.readNetFromDarknet(str(detector.DEFAULT_CFG))
    assert len(np.array(net.getUnconnectedOutLayers()).flatten()) == 3  # three YOLO heads
