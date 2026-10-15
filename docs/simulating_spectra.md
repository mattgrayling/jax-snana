(simulating-spectra)=
# Simulating spectra

jax-SNANA can simulate spectroscopic observations using an instrument
response that specifies the wavelength coverage, resolution and sensitivity
of the spectrograph. Exposure times can be specified directly or calculated
to achieve a target signal-to-noise ratio over a chosen wavelength interval.
Observations can be requested at particular phases or scheduled through the
survey SIMLIB, and host-galaxy light can also be included in the simulated
spectra.

## Requesting spectra

`SPECTROGRAPH_FILE` supplies the instrument response, tabulating the
wavelength bins, resolution and signal-to-noise at reference magnitudes
for a set of exposure times. The {ref}`spectrograph response format
<format-spectrograph>` describes these inputs.

The response can also be supplied through a `SPECTROGRAPH` entry in the
{ref}`calibration input <format-kcor>` or an existing kcor FITS file.

For example, to request spectra at peak and ten rest-frame days later,
each with an 1800-second exposure, add the following to the simulation
input. This uses a toy instrument response supplied with the code:

```text
SPECTROGRAPH_FILE: tests/fixtures/data/spectrograph/SPECTROGRAPH_TEST.DAT
TAKE_SPECTRUM: TREST(0:0) TEXPOSE_ZPOLY(1800,0)
TAKE_SPECTRUM: TREST(10:10) TEXPOSE_ZPOLY(1800,0)
```

`TREST(lo:hi)` draws one rest-frame phase in that interval for each event;
equal endpoints fix the phase. `TEXPOSE_ZPOLY` gives the exposure time in
seconds as a polynomial in redshift, starting at the constant term.

`examples/quickstart/sim_spectra.input` combines these observations with
the SALT simulation from {doc}`getting_started`.
After {ref}`preparing the examples <installation-examples>`, run it from
the repository root:

```bash
snlc_sim examples/quickstart/sim_spectra.input
```

To request a target signal-to-noise instead of a fixed exposure, replace a
`TAKE_SPECTRUM` line with, for example:

```text
TAKE_SPECTRUM: TREST(0:0) SNR_ZPOLY(10,0) SNR_LAMOBS(5000:6000)
```

This requests a combined signal-to-noise ratio of 10 across 5000–6000 Å
in the observer frame, within the example instrument's wavelength coverage.
Use `SNR_LAMREST` for a rest-frame interval. By default, the exposure is
limited to the range in the instrument response, so the target may not
be reached.

Spectra can also be scheduled at the SIMLIB's `SPECTROGRAPH:` epochs by
enabling `USE_SIMLIB_SPECTROGRAPH`. The calibration can define synthetic
passbands over selected wavelength intervals, with photometry generated
in those bands at the same epochs; see {ref}`format-kcor`.

Host-galaxy light can be added to the supernova spectrum using a
{ref}`host-galaxy catalogue (HOSTLIB) <simulating-hosts>` and host spectra
supplied through `HOSTLIB_SPECBASIS_FILE` or
`HOSTLIB_SPECDATA_FILE`. `TAKE_SPECTRUM_HOSTFRAC` sets the fraction of the
host spectrum added. {doc}`simulation_keys` lists the spectroscopic settings.

## Output

For this example, the spectra are written to
`QUICKSTART_SPECTRA/QUICKSTART_SPECTRA_SPEC.FITS.gz`.
The file contains a `SPECTRO_HEADER` table identifying each event and
exposure, and a `SPECTRO_FLUX` table with
wavelength bounds, noisy flux density, its uncertainty and the noise-free
model flux density. Wavelengths are in observer-frame angstroms and flux
densities are in erg s$^{-1}$ cm$^{-2}$ angstrom$^{-1}$. Use the header's
`PTRSPEC_MIN` and `PTRSPEC_MAX` to select a spectrum's rows; these are
inclusive, one-based indices. For example:

```python
from astropy.io import fits

with fits.open("QUICKSTART_SPECTRA/QUICKSTART_SPECTRA_SPEC.FITS.gz") as hdus:
    header = hdus["SPECTRO_HEADER"].data[0]
    start, stop = int(header["PTRSPEC_MIN"]) - 1, int(header["PTRSPEC_MAX"])
    spectrum = hdus["SPECTRO_FLUX"].data[start:stop].copy()

wavelength = (spectrum["LAMMIN"] + spectrum["LAMMAX"]) / 2
flux, flux_error = spectrum["FLAM"], spectrum["FLAMERR"]
```

The spectra accompany the light-curve output described under
{ref}`simulation-output`.
