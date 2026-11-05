
import numpy as np
import pytest

from modules.image_processing import (
    apply_grayscale,
    apply_brightness,
    apply_contrast,
    apply_sepia,
    process_frame,
    FILTERS
)


@pytest.fixture
def sample_image():
    return np.full((100, 100, 3), 120, dtype=np.uint8)


def test_grayscale(sample_image):
    result = apply_grayscale(sample_image)

    assert result.shape == sample_image.shape


def test_brightness(sample_image):
    result = apply_brightness(sample_image, 50)

    assert result.mean() > sample_image.mean()


def test_contrast(sample_image):
    result = apply_contrast(sample_image, 1.5)

    assert result.shape == sample_image.shape


def test_sepia(sample_image):
    result = apply_sepia(sample_image)

    assert result.shape == sample_image.shape


@pytest.mark.parametrize("filter_name", FILTERS)
def test_process_frame_returns_image(sample_image, filter_name):

    # Simulated blue replacement background
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    background[:] = [255, 0, 0]

    # Simulated frame with green background
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:] = [0, 255, 0]

    # Add a red subject
    frame[25:75, 25:75] = [0, 0, 255]

    result = process_frame(
        frame,
        background,
        filter_name
    )

    assert result.shape == frame.shape
    assert result.dtype == np.uint8
