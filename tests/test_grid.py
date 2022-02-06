from toolsvm.grid import GridOption, GridHyperParameter
from hypothesis import strategies as st, given
import pytest


@given(
    svm_type=st.integers(min_value=0, max_value=4),
    c_range=st.tuples(st.integers(), st.integers(), st.integers()),
    g_range=st.tuples(st.integers(), st.integers(), st.integers()),
    p_range=st.tuples(st.integers(), st.integers(), st.integers()),
    with_c=st.booleans(),
    with_g=st.booleans(),
    with_p=st.booleans(),
    fold=st.integers(),
    with_output=st.booleans(),
    nb_process=st.integers(),
)
def test_grid_options(**kwargs):
    r = None
    if not any(kwargs[f"with_{k}"] for k in "cgp"):
        r = pytest.raises(ValueError)
        r.__enter__()

    try:
        GridOption(dataset=".datasets/australian_scale.txt", out_pathname=None, resume_pathname=None, svm_options={}, **kwargs)
    except Exception as e:
        if r:
            assert isinstance(e, r.expected_exception)
            r.__exit__(e.__class__, e, e.__traceback__)

    assert True


@given(c=st.floats(max_value=1023), g=st.floats(max_value=1023), p=st.floats(max_value=1023))
def test_hyperparameters(c, g, p):
    hp = GridHyperParameter(c, g, p)
    assert hp.c == 2**hp.log2c
    assert hp.p == 2**hp.log2p
    assert hp.g == 2**hp.log2g
    assert hp == hp
    assert hash(hp) == hash((hp.log2c, hp.log2g, hp.log2p))
