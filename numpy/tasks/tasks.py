import numpy as np

"""
TODO
Implement the functions below using only numpy operations and functions.
DO NOT use any Python control flow (no cycles, no ifs, etc.) in any of the tasks.
"""


def max_row_index(x: np.ndarray) -> int:
    """
    Find the index of the row that contains the maximum value in `x`.
    You can assume that `x` is a 2D array and that there is only one maximal value.

    See tests for examples.
    """


def sum_edges(x: np.ndarray) -> int:
    """
    Sum the edges of the input array.
    The input will be a 2D array of arbitrary shape (each dimension will have at least a single element).

    DO NOT count any edge element more than once!
    See tests for examples.
    """


def to_bw(img: np.ndarray) -> np.ndarray:
    """
    Convert the input np.uint8 RGB image to a black-and-white RGB image.
    The returned image should have shape (height, image, 3) and data type np.uint8.

    See data/fei.png -> data/fei-bg.png.
    """


def mirror(img: np.ndarray) -> np.ndarray:
    """
    Mirror the input np.uint8 RGB image along the vertical axis.

    See data/pyladies.png -> data/pyladies-mirrored.png.
    """


def split_and_rotate(img: np.ndarray, n: int) -> np.ndarray:
    """
    Split the input np.uint8 RGB image into `n` parts horizontally.
    Then place the parts so that they lie below each other vertically.

    See data/pyladies.png -> data/pyladies-split-and-rotate.png.
    """


def apply_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply a binary mask to the input np.uint8 RGB image.
    The mask will have the same shape as the input image.

    All pixels in `img` that are not white in `mask` should be set to black color.

    See data/geralt.png (image) + data/geralt-mask.png (mask) -> data/geralt-masked.png.
    """


def blend(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend the two input np.uint8 RGB images together using alpha blending.
    The `alpha` parameter will be in the interval [0.0, 1.0].
    It specifies how visible should `img_a` be.

    The returned image should have data type np.uint8.

    See data/geralt-a.png (img_a) + data/geralt-b.png (img_a) -> data/geralt-blend-{alpha}.png.
    """
