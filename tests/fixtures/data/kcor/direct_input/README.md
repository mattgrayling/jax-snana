# Direct calibration input fixture

The recorded FITS files were generated with the local SNANA `kcor.exe` on 9 October
2026, using `kcor.exe <input> SKIPKCOR`. The input and all of its
spectra and transmission curves are included, so the regression tests need
neither SNANA nor its data tree.

The reference spectra are synthetic: `ab.dat` is flat at 3631 Jy and
`bd17.dat` is a power law, labelled BD17 to exercise the reference-standard
mapping. They are test spectra, not survey calibration standards. The four
triangular filters cover COUNT and ENERGY responses, AB and BD17 standards,
conversions in both directions, primary-magnitude sums and `ZPOFF.DAT`.

FITS columns are stored in single precision; comparisons allow their
rounding error while reading the input in double precision.

`calibration_sed.input` supplies a synthetic spectral time series to check
its 20 Å calibration grid. `calibration_cut.input` applies `LEAKAGE_CUT`
before interpolation onto the stored grid. The FITS recordings are gzipped.

`calibration_duplicate.input` adds wavelength-shifted passbands.
`calibration_spectrograph.input` adds a spectrograph calibration and two
synthetic bands; `calibration_rebin.input` combines its wavelength bins in
pairs before solving the calibration. These recordings check transmissions,
reference fluxes and the spectrograph noise calibration.

`calibration_spectrograph_duplicate.input` checks wavelength shifts of the
synthetic bands, including their integration bounds.

The three `calibration_oob*` cases use `FILTER_OOB` on all four bands,
including COUNT/ENERGY responses and magnitude-system conversion. They
also cover `LEAKAGE_CUT` and a model-defined calibration grid. Their FITS
references were generated from the corresponding `.reference.input` files,
which name explicit leakage curves rather than using `FILTER_OOB`.

For each original curve, `*_oob.dat` adds 0.001 times the original peak to
its transmission samples and extends each edge towards 3007–8993 Å in
whole steps of the original edge spacing. The non-aligned bounds exercise
edge rounding. Extended samples contain just the constant leakage.
These curves were constructed independently of the calibration loader.

The local SNANA executable deliberately stops on non-zero `FILTER_OOB`,
so these recordings validate the calibration of the equivalent explicit
curves; they do not constitute native `FILTER_OOB` execution parity.
