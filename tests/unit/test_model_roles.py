"""Registering a model must not make unrelated consumers depend on its roles."""

import subprocess
import sys
import textwrap


def test_fit_only_model_registration(repo_root):
    # Import-time registry scans must run after registration, in a fresh process.
    code = textwrap.dedent("""
        from jax_snlc_core.models import MODEL_REGISTRY, SNModel, FittedModel

        class FitOnly(SNModel, FittedModel):
            GENMODEL_ALIASES = ('fitonly',)

        MODEL_REGISTRY['fitonly'] = FitOnly
        import jax_pipeline
        from jax_snlc_sim.config_parser import parse_snana_input_file, build
        from jax_snlc_core.utils.error_handling import SNANAError

        parsed = parse_snana_input_file(
            'tests/fixtures/snana/atmosphere_replay/input.input')
        assert parsed.model.model_type == 'salt'
        assert 'x1' in parsed.model.populations
        try:
            build({'GENMODEL': 'fitonly', 'GENFILTERS': 'g'})
        except SNANAError as error:
            assert 'SimulatedModel role' in str(error)
        else:
            raise AssertionError('a fit-only model was accepted for simulation')
    """)
    result = subprocess.run([sys.executable, '-c', code], cwd=repo_root,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
