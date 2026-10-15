"""The fitter's view of ``$SNDATA_ROOT/SURVEY.DEF``.

Both the loader (FIELD names) and the table writers (IDSURVEY) need survey
metadata, so it lives with neither. The file itself is read by
``jax_snlc_core.io.survey_def``; what belongs here is the fitter's answer for a
survey the file does not name, which is a missing value rather than an abort.
"""

import numpy as np

from jax_snlc_core.io.survey_def import (load_survey_fields,
                                         load_survey_ids)


def survey_id(name):
    """IDSURVEY from $SNDATA_ROOT/SURVEY.DEF; -9 when unmatched."""
    return load_survey_ids()[0].get(name, -9)


def event_survey_ids(survey, subsurveys, n):
    """The IDSURVEY column: each event's SUBSURVEY resolved through
    SURVEY.DEF where the HEAD names one, else the survey's own id
    (snana.F90:3839-3843, :12620; written under the name IDSURVEY,
    :23294-23297)."""
    ids = np.full(n, survey_id(survey))
    if subsurveys is None:
        return ids
    names = np.char.strip(np.asarray(subsurveys, dtype='U'))
    for name in np.unique(names[names != '']):
        ids[names == name] = survey_id(name)
    return ids


def survey_fields(name):
    """The survey's FIELD names from $SNDATA_ROOT/SURVEY.DEF, in file order."""
    return load_survey_fields(name)
