"""Explicit prefix + prepared HSL merge composition, not full Adobe Prepare."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import native_renderer as native


@pytest.mark.parametrize('flag', [0, 1, 2, 128, 255])
@pytest.mark.parametrize('scalar', [0., .35, 1., 1.75, 2.])
def test_flag_prefix_then_hsl_merge_and_pixels(flag, scalar):
    current = np.zeros(132, np.int32)
    current[43:67] = np.arange(-96, 96, 8, dtype=np.int32)
    current[129] = -999999
    header = np.ones(16, np.uint8)
    header[7] = flag
    saved = current.copy(), header.copy()
    reset, mask = native.native_acr_basic_flag_slots(current, header)
    expected = current[43:67] if flag else np.zeros(24, np.int32)
    np.testing.assert_array_equal(reset[43:67], expected)
    assert reset[129] == -999999  # This prefix is not the later sentinel pass.
    assert all(bool(mask[i//32] & (1 << (i%32))) == (flag == 0 and 43 <= i <= 66)
               for i in range(132))
    defaults = np.arange(92, -100, -8, dtype=np.int32)
    merged, wrote, _ = native.native_acr_profile_hsl_controls(expected, defaults, scalar)
    after, after_wrote, _ = native.native_acr_profile_hsl_controls(reset[43:67], defaults, scalar)
    np.testing.assert_array_equal(after, merged)
    assert after_wrote == wrote == (1 << 24)-1
    rgb = np.random.default_rng(129).uniform(.01,.99,(3,17,3)).astype(np.float32)
    actual = native.native_acr_profile_hsl_sdr(rgb, np.r_[reset[43:67],0.], defaults, scalar)
    reference = native.native_acr_hsl_sdr(rgb, np.r_[merged,0.])
    np.testing.assert_array_equal(actual.view(np.uint32), reference.view(np.uint32))
    np.testing.assert_array_equal(current, saved[0])
    np.testing.assert_array_equal(header, saved[1])


def test_reset_hsl_does_not_suppress_separate_look_defaults():
    current = np.full(132, 50, np.int32)
    header = np.ones(16, np.uint8)
    header[7] = 0
    reset, _ = native.native_acr_basic_flag_slots(current, header)
    defaults = np.zeros(24, np.int32)
    defaults[0] = 50
    merged, wrote, _ = native.native_acr_profile_hsl_controls(reset[43:67], defaults, 1.)
    assert reset[43] == 0 and merged[0] == 50 and wrote & 1
    # Clearing the current group does not prove absent Look/default effects.
