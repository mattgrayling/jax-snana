"""SNANA point-source limiting-magnitude calculator (MAGLIMIT_calculator,
sntools.c:5259; Belov & Glazov 2010)."""

import math

from jax_snlc_core.utils.error_handling import SNANAError


def maglimit_calculator(zpt: float, psf: float, skymag: float,
                        snr: float) -> float:
    """Point-source mag at signal-to-noise ``snr``.

    ``zpt`` is the zero point in photoelectrons (``Npe =
    10**[-0.4*(mag-ZPT)]``); ``psf`` is the PSF FWHM in arcsec; ``skymag``
    is the sky brightness in mag/arcsec^2.
    """
    if zpt <= 1.01 or psf <= 0.001 or skymag <= 1.0:
        return 0.0

    ea = 1.00      # fraction of source flux
    omega = 1.51   # effective sky-noise area factor
    acc = 0.001
    t2 = 5.0 * math.log10(snr / ea)
    t_area = (omega * psf) ** 2
    t1 = 2.5 * math.log10(t_area)

    mlim = 20.0
    eps = 10000.0
    cnt = 100
    first = True
    while eps > acc and cnt > 0:
        t_mlim = 0.5 * (zpt + skymag - t1 - t2) if first else mlim
        first = False
        arg = 0.4 * (skymag - t_mlim)
        mlim = 0.5 * (t_mlim + zpt - t2) - 1.25 * math.log10(1.0 + t_area * 10.0 ** (-arg))
        eps = abs(mlim - t_mlim)
        cnt -= 1
    if cnt == 0:
        raise SNANAError(
            f"MLIM={mlim:.3f} value has not converged: eps={eps:f} "
            f"(ZP={zpt:.3f} PSF={psf:.3f} SKYMAG={skymag:.2f} SNR={snr:.1f})",
            function_name="maglimit_calculator")
    return mlim
