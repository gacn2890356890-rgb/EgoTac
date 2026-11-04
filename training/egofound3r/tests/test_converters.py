"""Each converter against a synthetic copy of its dataset's documented layout, with a stand-in MANO.

This checks parsing, frames, and units. It cannot show that the documented layout matches the real files.
"""
from __future__ import annotations

import io
import json
import pickle
import subprocess
import sys
import tarfile

import cv2
import numpy as np
import pytest
import trimesh

from egotac.data.mano import _PARENTS
from egotac.data.source import hand_rate_indices, window_starts
from egotac.data.splits import plan_splits

WRIST = np.array([0.0, 0.0, 0.0])
TRANS = np.array([0.1, -0.05, 0.4])


@pytest.fixture
def mano_dir(tmp_path):
    folder = tmp_path / "mano"
    folder.mkdir()
    rng = np.random.default_rng(0)
    regressor = np.zeros((16, 778))
    regressor[:, 0] = 1.0  # every rest joint at vertex 0, which sits at the origin
    template = rng.normal(size=(778, 3)) * 0.02
    template[0] = WRIST
    weights = np.zeros((778, 16))
    weights[:, 0] = 1.0
    for side in ("LEFT", "RIGHT"):
        data = {
            "kintree_table": np.array([[2**32 - 1] + list(_PARENTS[1:]), list(range(16))]),
            "v_template": template,
            "shapedirs": np.zeros((778, 3, 10)),
            "posedirs": np.zeros((778, 3, 135)),
            "J_regressor": regressor,
            "weights": weights,
            "f": np.array([[0, 1, 2], [1, 2, 3]]),
            "hands_mean": np.zeros(45),
            "hands_components": np.eye(45),
        }
        (folder / f"MANO_{side}.pkl").write_bytes(pickle.dumps(data))
    return folder


def write_mesh(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(path)


def image(height=48, width=64):
    return np.full((height, width, 3), 128, dtype=np.uint8)


def convert(module, root, output, mano_dir, *extra):
    subprocess.run([sys.executable, "-m", f"egotac.data.{module}", "--root", str(root), "--output", str(output), "--mano-dir", str(mano_dir), *extra], check=True)
    rows = [json.loads(line) for line in (output / "manifest.jsonl").open()]
    assert rows
    with np.load(output / rows[0]["path"]) as data:
        return rows[0], {key: data[key] for key in data.files}


def test_window_layout_matches_the_official_rounding():
    assert hand_rate_indices(0, 30.0)[:3] == [0, 3, 6] and hand_rate_indices(0, 30.0)[-1] == 60
    assert hand_rate_indices(0, 15.0)[:4] == [0, 2, 3, 4] and hand_rate_indices(0, 15.0)[-1] == 30
    assert window_starts(61, 30.0, 21) == [0]
    assert window_starts(60, 30.0, 21) == []


def test_fixed_test_split_keeps_h2o_subject_four_out_of_training():
    names = [f"subject{s}/h1/{n}" for s in (1, 4) for n in range(3)]
    plan = plan_splits("h2o", names, lambda name: name.startswith("subject4/"))
    assert {name for name, split in plan.items() if split == "test"} == {f"subject4/h1/{n}" for n in range(3)}


def test_h2o(tmp_path, mano_dir):
    root = tmp_path / "h2o"
    cam = root / "subject1/h1/0/cam4"
    for name in ("rgb", "cam_pose", "hand_pose_MANO", "obj_pose_RT"):
        (cam / name).mkdir(parents=True)
    (cam / "cam_intrinsics.txt").write_text("100 100 32 24 64 48\n")
    hand = np.zeros(59)
    hand[0], hand[1:4] = 1, TRANS
    no_hand = np.zeros(59)
    pose = np.r_[1, np.eye(4).ravel()]
    for t in range(62):
        name = f"{t:06d}"
        cv2.imwrite(str(cam / "rgb" / f"{name}.png"), image())
        np.savetxt(cam / "cam_pose" / f"{name}.txt", np.eye(4).ravel())
        np.savetxt(cam / "hand_pose_MANO" / f"{name}.txt", np.r_[no_hand, hand])
        np.savetxt(cam / "obj_pose_RT" / f"{name}.txt", pose)
    write_mesh(tmp_path / "objects/book.obj")
    row, clip = convert("convert_h2o", root, tmp_path / "out", mano_dir, "--object-dir", str(tmp_path / "objects"))
    assert (row["dataset"], row["split"]) == ("h2o", "train")
    assert not clip["hand_valid"][0, 0] and clip["hand_valid"][0, 1]
    assert np.allclose(clip["joints_cam"][0, 1, 0], TRANS, atol=1e-5)
    assert np.isfinite(clip["dist_joints"][0, 1]).all()


def test_hoi4d(tmp_path, mano_dir):
    root = tmp_path / "hoi4d"
    sequence = "ZY1/H1/C2/N7/S1/s1/T1"
    folder = root / "HOI4D_release" / sequence
    (folder / "align_rgb").mkdir(parents=True)
    (folder / "objpose").mkdir()
    (folder / "3Dseg").mkdir()
    writer = cv2.VideoWriter(str(folder / "align_rgb/image.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for _ in range(40):
        writer.write(image())
    writer.release()
    (folder / "3Dseg/output.log").write_text("".join("0 0 1\n1 0 0 0\n0 1 0 0\n0 0 1 0\n0 0 0 1\n" for _ in range(40)))
    (root / "camera_params/ZY1").mkdir(parents=True)
    np.save(root / "camera_params/ZY1/intrin.npy", np.array([[100.0, 0, 32], [0, 100, 24], [0, 0, 1]]))
    write_mesh(root / "HOI4D_CAD_Model_for_release/rigid/Mug/007.obj")
    hands = root / "handpose/refinehandpose_right" / sequence
    hands.mkdir(parents=True)
    (root / "handpose/refinehandpose_left").mkdir()
    for t in range(40):
        (hands / f"{t}.pickle").write_bytes(pickle.dumps({"poseCoeff": np.zeros(48), "beta": np.zeros(10), "trans": TRANS}))
        (folder / "objpose" / f"{t}.json").write_text(json.dumps({"dataList": [{"center": {"x": 0, "y": 0, "z": 0.4}, "rotation": {"x": 0, "y": 0, "z": 0}, "dimensions": {"length": 1, "width": 1, "height": 1}}]}))
    row, clip = convert("convert_hoi4d", root, tmp_path / "out", mano_dir)
    assert row["dataset"] == "hoi4d" and row["sequence"] == sequence
    assert not clip["hand_valid"][0, 0] and clip["hand_valid"][0, 1]
    assert np.allclose(clip["joints_cam"][0, 1, 0], TRANS, atol=1e-5)
    assert np.isfinite(clip["dist_joints"][0, 1]).all()


def test_arctic(tmp_path, mano_dir):
    root = tmp_path / "arctic"
    (root / "raw_seqs/s01").mkdir(parents=True)
    images = root / "images/s01/box_grab_01/0"
    images.mkdir(parents=True)
    frames = 62
    side = {"rot": np.zeros((frames, 3)), "pose": np.zeros((frames, 45)), "trans": np.tile(TRANS, (frames, 1)), "shape": np.zeros(10)}
    np.save(root / "raw_seqs/s01/box_grab_01.mano.npy", {"left": side, "right": side}, allow_pickle=True)
    params = np.zeros((frames, 7))
    params[:, 6] = 400.0  # millimeters
    np.save(root / "raw_seqs/s01/box_grab_01.object.npy", params)
    shift = np.array([0.0, 0.0, 0.1])
    camera = {"R_k_cam_np": np.tile(np.eye(3), (frames, 1, 1)), "T_k_cam_np": np.tile(shift, (frames, 1)), "intrinsics": np.array([[100.0, 0, 32], [0, 100, 24], [0, 0, 1]]), "dist8": np.zeros(8)}
    np.save(root / "raw_seqs/s01/box_grab_01.egocam.dist.npy", camera, allow_pickle=True)
    template = root / "meta/object_vtemplates/box"
    write_mesh(template / "mesh.obj")
    (template / "parts.json").write_text(json.dumps([0] * 8))
    (root / "meta/misc.json").write_text(json.dumps({"s01": {"ioi_offset": 5}}))
    for t in range(frames):
        cv2.imwrite(str(images / f"{t + 5:05d}.jpg"), image())
    row, clip = convert("convert_arctic", root, tmp_path / "out", mano_dir)
    assert row["dataset"] == "arctic"
    assert clip["hand_valid"][0].all()
    assert np.allclose(clip["joints_cam"][0, 1, 0], TRANS + shift, atol=1e-5)
    # The box surface sits 0.05 m from its center at z = 0.5 in the camera frame, near the hand at 0.5.
    assert clip["dist_joints"][0, 1, 0] < 0.2


def test_arctic_articulation_turns_only_the_top_part():
    from egotac.data.convert_arctic import articulated_tris

    vertices = np.array([[1000.0, 0, 0], [0, 1000.0, 0], [0, 0, 1000.0]])
    faces = np.array([[0, 1, 2]])
    params = np.r_[np.pi / 2, np.zeros(3), np.zeros(3)]
    tris = articulated_tris(vertices, faces, np.array([True, False, False]), params)
    assert np.allclose(tris[0, 0], [0, -1, 0], atol=1e-5)  # rotated about -z, in meters
    assert np.allclose(tris[0, 1], [0, 1, 0], atol=1e-5)


def test_hot3d(tmp_path, mano_dir):
    root = tmp_path / "hot3d"
    (root / "train_aria").mkdir(parents=True)
    (root / "object_models_eval").mkdir()
    trimesh.creation.box(extents=(0.1, 0.1, 0.1)).export(root / "object_models_eval/obj_000001.glb")
    (root / "clip_definitions.json").write_text(json.dumps({"7": {"sequence_id": "P0001_x", "device": "Aria"}}))
    (root / "clip_splits.json").write_text(json.dumps({"train": {"Aria": [7]}}))
    identity = {"translation_xyz": [0, 0, 0], "quaternion_wxyz": [1, 0, 0, 0]}
    camera = {"214-1": {"calibration": {"label": "rgb", "stream_id": "214-1", "serial_number": "x", "image_width": 64, "image_height": 48, "projection_model_type": "CameraModelType.FISHEYE624", "projection_params": [100.0, 32.0, 24.0] + [0.0] * 12, "T_device_from_camera": identity, "max_solid_angle": 1.5}, "T_world_from_camera": identity}}
    hand = {"mano_pose": {"thetas": [0.0] * 15, "wrist_xform": [0.0, 0.0, 0.0, *TRANS]}}
    objects = {"1": [{"object_bop_id": 1, "T_world_from_object": {"translation_xyz": [0, 0, 0.4], "quaternion_wxyz": [1, 0, 0, 0]}}]}

    def add(tar, name, payload):
        info = tarfile.TarInfo(name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    with tarfile.open(root / "train_aria/clip-000007.tar", "w") as tar:
        add(tar, "__hand_shapes.json__", json.dumps({"mano": [0.0] * 10}).encode())
        for t in range(62):
            key = f"{t:06d}"
            add(tar, f"{key}.image_214-1.jpg", cv2.imencode(".jpg", image())[1].tobytes())
            add(tar, f"{key}.cameras.json", json.dumps(camera).encode())
            add(tar, f"{key}.hands.json", json.dumps({"right": hand}).encode())
            add(tar, f"{key}.objects.json", json.dumps(objects).encode())
    row, clip = convert("convert_hot3d", root, tmp_path / "out", mano_dir)
    assert (row["dataset"], row["sequence"]) == ("hot3d_aria", "P0001_x")
    assert not clip["hand_valid"][0, 0] and clip["hand_valid"][0, 1]
    assert np.allclose(clip["joints_cam"][0, 1, 0], TRANS, atol=1e-5)
