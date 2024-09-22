import contextlib
from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


@contextlib.contextmanager
def temp_seed(seed):
    state: Tuple[str, np.ndarray, int, int, float] = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


def show_image_and_gt(image, objects, fig_num=None) -> Axes:
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objects is not None:
        for polygon in objects:
            poly: np.ndarray = np.array(polygon['polygon'])[list(np.arange(len(polygon['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=polygon['label'])
            labels.add(polygon['label'])
        if len(labels) > 1:
            plt.legend()
    return plt.gca()
