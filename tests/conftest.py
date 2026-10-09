import pytest

from stackxai.data import load_wbcd


@pytest.fixture(scope="session")
def wbcd():
    return load_wbcd()
