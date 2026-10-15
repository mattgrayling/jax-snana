"""Spectrograph calibration, SNR computation and TAKE_SPECTRUM generation."""

from jax_snlc_core.photometry.spectrograph import (
    SpectrographCalibration,
    read_spectrograph,
    get_snr,
    get_snr_from_calib,
)
from jax_snlc_sim.spectrograph.genspec import (
    SpectrumResult,
    build_smearing_matrix,
    generate_all_spectra,
    solve_texpose_batch,
    snrsum_rest_v_sq,
)
