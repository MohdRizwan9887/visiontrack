import numpy as np

from visiontrack.tracker import SORTTracker, iou_matrix


def test_iou_identity_and_disjoint():
    a = np.array([[0, 0, 10, 10]], dtype=float)
    assert np.isclose(iou_matrix(a, a)[0, 0], 1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], dtype=float))[0, 0] == 0.0


def test_track_id_persistence_with_motion():
    SORTTracker.reset_ids()
    trk = SORTTracker(min_hits=2)
    ids = set()
    for t in range(30):
        # two objects moving at different velocities
        d = np.array([[10 + 4 * t, 50, 60 + 4 * t, 120, 0.9, 0],
                      [400 - 3 * t, 200, 460 - 3 * t, 300, 0.8, 2]], dtype=np.float32)
        out = trk.update(d)
        ids |= set(out[:, 4].astype(int))
    assert ids == {1, 2}


def test_track_survives_missed_detections():
    SORTTracker.reset_ids()
    trk = SORTTracker(min_hits=2, max_age=10)
    ids = set()
    for t in range(25):
        d = np.array([[10 + 5 * t, 50, 70 + 5 * t, 130, 0.9, 0]], dtype=np.float32)
        if 10 <= t < 14:  # detector misses the object for 4 frames
            d = np.zeros((0, 6), dtype=np.float32)
        out = trk.update(d)
        ids |= set(out[:, 4].astype(int))
    assert ids == {1}
