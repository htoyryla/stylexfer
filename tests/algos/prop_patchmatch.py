# Neural Imagen — Copyright (c) 2018, Alex J. Champandard. Code licensed under the GNU AGPLv3.

import torch
from imagen.algos import patchmatch

import pytest
from hypothesis import given, event, strategies as H


def make_square_tensor(size, channels):
    return torch.rand((1, channels, size, size), dtype=torch.float)


def Tensor(range=(1, 32), channels=None) -> H.SearchStrategy[torch.Tensor]:
    return H.builds(
        make_square_tensor,
        size=H.integers(min_value=range[0], max_value=range[-1]),
        channels=H.integers(min_value=channels or 1, max_value=channels or 8),
    )


Coord = H.tuples(H.integers(), H.integers())
CoordList = H.lists(Coord, min_size=1, max_size=32)


@given(content=Tensor(channels=4), style=Tensor(channels=4))
def test_indices_random_range(content, style):
    """Determine that random indices are in range.
    """
    pm = patchmatch.PatchMatcher(content, style, indices="random")
    assert pm.indices[:, 0, :, :].min() >= 0
    assert pm.indices[:, 0, :, :].max() < style.shape[2]

    assert pm.indices[:, 1, :, :].min() >= 0
    assert pm.indices[:, 1, :, :].max() < style.shape[3]


@given(content=Tensor(channels=5), style=Tensor(channels=5))
def test_indices_linear_range(content, style):
    """Determine that random indices are in range.
    """
    pm = patchmatch.PatchMatcher(content, style, indices="linear")
    assert pm.indices[:, 0, :, :].min() >= 0
    assert pm.indices[:, 0, :, :].max() < style.shape[2]

    assert pm.indices[:, 1, :, :].min() >= 0
    assert pm.indices[:, 1, :, :].max() < style.shape[3]


@given(content=Tensor(channels=3), style=Tensor(channels=3))
def test_scores_range(content, style):
    """Determine that the scores of random patches are in correct range.
    """
    pm = patchmatch.PatchMatcher(content, style)
    assert pm.scores.min() >= 0.0
    assert pm.scores.max() <= 1.0


@given(
    content=Tensor(range=(4, 32), channels=3), style=Tensor(range=(4, 32), channels=3)
)
def test_indices_random(content, style):
    """Determine that random indices are indeed random in larger grids.
    """
    pm = patchmatch.PatchMatcher(content, style)
    assert pm.indices.min() != pm.indices.max()


@given(array=Tensor(range=(2, 8)))
def test_indices_linear(array):
    """Indices of the indentity transformation should be linear.
    """
    pm = patchmatch.PatchMatcher(array, array, indices="linear")
    assert (
        pm.indices[:, 0, :, :]
        == torch.arange(start=0, end=array.shape[2]).view(1, -1, 1)
    ).all()
    assert (
        pm.indices[:, 1, :, :]
        == torch.arange(start=0, end=array.shape[3]).view(1, 1, -1)
    ).all()


@given(array=Tensor(range=(2, 16)))
def test_scores_identity(array):
    """The score of the identity operation with linear indices should be one.
    """
    pm = patchmatch.PatchMatcher(array, array, indices="linear")
    assert pytest.approx(1.0) == pm.scores.min()


@given(
    content=Tensor(range=(4, 16), channels=2), style=Tensor(range=(4, 16), channels=2)
)
def test_scores_zero(content, style):
    """Scores must be zero if inputs vary on different dimensions.
    """
    content[:, :, 0], style[:, :, 1] = 0.0, 0.0
    pm = patchmatch.PatchMatcher(content, style)
    assert pytest.approx(0.0) == pm.scores.max()


@given(
    content=Tensor(range=(4, 16), channels=2), style=Tensor(range=(4, 16), channels=2)
)
def test_scores_one(content, style):
    """Scores must be one if inputs only vary on one dimension.
    """
    content[:, 0, :, :], style[:, 0, :, :] = 0.0, 0.0
    pm = patchmatch.PatchMatcher(content, style)
    assert pytest.approx(1.0) == pm.scores.min()


@given(
    content=Tensor(range=(4, 32), channels=2), style=Tensor(range=(4, 32), channels=2)
)
def test_scores_zero(content, style):
    """Scores must be zero if inputs vary on different dimensions.
    """
    content[:, 0, :, :], style[:, 1, :, :] = 0.0, 0.0
    pm = patchmatch.PatchMatcher(content, style)
    assert pytest.approx(0.0) == pm.scores.max()


@given(
    content=Tensor(range=(4, 16), channels=5), style=Tensor(range=(4, 16), channels=5)
)
def test_scores_improve(content, style):
    """Scores must be one if inputs only vary on one dimension.
    """
    pm = patchmatch.PatchMatcher(content, style)
    before = pm.scores.sum()
    pm.search_patches_random(times=1)
    after = pm.scores.sum()
    event("equal? %i" % int(after == before))
    assert after >= before


@given(array=Tensor(range=(2, 8), channels=5))
def test_propagate_down_right(array):
    """Propagating the identity transformation expects indices to propagate
    one cell at a time, this time down and towards the right. 
    """
    pm = patchmatch.PatchMatcher(array, array, indices="zero")

    pm.search_patches_propagate(steps=[1])
    assert (pm.indices[:, :, 1, 0] == torch.tensor([1, 0], dtype=torch.long)).all()
    assert (pm.indices[:, :, 0, 1] == torch.tensor([0, 1], dtype=torch.long)).all()

    pm.search_patches_propagate(steps=[1])
    assert (pm.indices[:, :, 1, 1] == torch.tensor([1, 1], dtype=torch.long)).all()


@given(array=Tensor(range=(2, 8), channels=5))
def test_propagate_up_left(array):
    """Propagating the identity transformation expects indices to propagate
    one cell at a time, here up and towards the left.
    """
    y, x = array.shape[-2:]
    pm = patchmatch.PatchMatcher(array, array, indices="zero")
    pm.indices[:, 0, -1, -1] = y - 1
    pm.indices[:, 1, -1, -1] = x - 1
    pm.improve_patches(pm.indices)

    pm.search_patches_propagate(steps=[1])
    assert (
        pm.indices[:, :, y - 2, x - 1] == torch.tensor([y - 2, x - 1], dtype=torch.long)
    ).all()
    assert (
        pm.indices[:, :, y - 1, x - 2] == torch.tensor([y - 1, x - 2], dtype=torch.long)
    ).all()

    pm.search_patches_propagate(steps=[1])
    assert (
        pm.indices[:, :, y - 2, x - 2] == torch.tensor([y - 2, x - 2], dtype=torch.long)
    ).all()
