import io

import pandas._testing as tm
import warnings
import pytest
import pandas as pd

def test_me():
    tm.assert_frame_equal(pd.DataFrame({'a': [1,2,3]}), pd.DataFrame({'a': [1,2, 4]}))
