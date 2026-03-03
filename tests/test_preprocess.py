import numpy as np
import os
from src.preprocess.face_preprocess import preprocess_image
from src.config import IMAGE_SIZE


SAMPLE_IMAGE = os.path.join("tests", "assets", "sample.jpg")


def test_preprocess_returns_vector():
    vec = preprocess_image(SAMPLE_IMAGE)

    # check conatiner type
    assert isinstance(vec, np.ndarray)

    # check shape
    assert vec.shape == (IMAGE_SIZE * IMAGE_SIZE,)

    # check data dtype
    assert vec.dtype == np.float32

    # check normalization range
    assert float(vec.min()) >= 0.0
    assert float(vec.max()) <= 1.0