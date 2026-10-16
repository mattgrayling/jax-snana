# jax-snana runtime image (SALT / BayeSN light-curve fitting).
#
# Installs jax-snana from this repo; the SALT/BayeSN models and the built-in
# filter calibration ship inside the package (jax_snlc_core/data), so nothing is
# downloaded at build or run time and no KCOR file is needed. `requests` is added
# for the OSG wrapper's result upload. A SALT fit is a least-squares, so CPU is
# fine and no GPU variant is needed.
FROM python:3.12-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

RUN python -m pip install --upgrade pip \
    && pip install . requests

# Fail the build if the runtime isn't importable, and warm the model/filter data
# path so a read-only worker never has to resolve it.
RUN python -c "from jax_snlc_fit import LightCurveFitter; \
    f = LightCurveFitter('SALT3.DOVEKIE', batch_size=1); \
    assert 'p48r' in f.calibration.get_filter_names(); \
    print('jax-snana runtime OK (built-in ZTF filters present)')"

CMD ["python"]
