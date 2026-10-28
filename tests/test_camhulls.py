"""The landing cams' hull pictures: every transparent PNG in src/app/cam_hulls/ is offered to the page, with the height of its hull's top."""
import numpy as np
from PIL import Image

from src.app import camhulls


def test_the_shipped_pictures_are_listed_with_a_sensible_apex():
    names = {h["name"]: h for h in camhulls.hull_list()}
    assert len(names) >= 4 and all(0.2 < h["apex"] < 0.8 and h["url"] == "/cam_hulls/" + h["name"] for h in names.values())


def test_apex_is_where_the_hull_fills_the_middle(tmp_path):
    a = np.zeros((100, 160, 4), np.uint8)
    a[60:, :, 3] = 255                                             # a hull whose top is at 60% of the height
    a[40:60, 78:82, 3] = 255                                       # and a thin leg above it, which must not count
    Image.fromarray(a, "RGBA").save(tmp_path / "x.png")
    assert abs(camhulls.apex_of(tmp_path / "x.png") - 0.6) < 0.02


def test_only_plain_png_names_inside_the_folder_are_served():
    assert camhulls.hull_file("hull_1.png") is not None
    for bad in ("../server.py", "..%2Fserver.py", "hull_1.png/../../x.png", "nothing.png", "hull_1.PNG.exe", ""):
        assert camhulls.hull_file(bad) is None


def test_the_airflow_stage_picture_is_served_by_plain_name_only():
    assert camhulls.model_file("airflow_stage.png") is not None
    for bad in ("../server.py", "..%2Fserver.py", "nothing.png", "airflow_stage.png/../x.png", ""):
        assert camhulls.model_file(bad) is None
