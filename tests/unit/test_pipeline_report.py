"""Reports use published fit results, including blinded parameter values."""
from jax_pipeline import BBC, Cosmology, Covariance, Fit, cosmology_summary


def test_summary_keeps_distinct_fits_and_blinded_values(tmp_path):
    fit = Fit("fit", fitres=tmp_path / "external.FITRES")
    bbc = BBC("bbc", data=fit, realisation=2, split_count=3, split_index=1)
    stage = Cosmology("wfit", covariance=Covariance("cov", distances=bbc))
    result = dict(success=False, blinded=True, method="grid", point_estimate_kind="mean",
                  parameter_order=["w0"], point_estimate={"w0": 42.},
                  parameters={"w0": dict(sigma=.2, q16=41.8, q84=42.2)},
                  fit=dict(chi2_total=10., n_rows=15, ndof_hd=13),
                  input=dict(simulation_truth={"w0": -1.}))
    frame = cosmology_summary([stage], {"wfit": {"ALL": result, "STAT": result}})
    assert list(frame.covopt) == ["ALL", "STAT"]
    assert list(frame.w0) == [42., 42.]
    assert frame.blinded.all() and not frame.success.any()
    assert list(frame.realisation) == [2, 2] and list(frame.split_count) == [3, 3]
    assert not any("truth" in name or "bias" in name for name in frame.columns)
