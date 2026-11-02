import pandas as pd
import pytest

from accepted_review_measurement import FIELDS, UNCERTAIN, agreed_only, candidate_fields, reliability


def labels(command):
    row = dict(review_id="synthetic", window="r5", test_command_status=command,
               execution_status="unknown", test_output_status="unknown_output", process_status="unknown",
               execution_detail="unknown", output_composition="unknown_output", runner_family="")
    return pd.DataFrame([row, dict(row, window="r15")])


def test_disagreement_does_not_choose_either_reviewer():
    a, b = labels("definite_test_command"), labels("definite_no_test_command")
    out = agreed_only(a, b)
    assert out.test_command_status.eq("ambiguous").all()
    assert out.unadjudicated_field_count.eq(1).all()
    assert out.itemwise_adjudicated.eq(False).all()
    assert a.test_command_status.eq("definite_test_command").all()
    assert b.test_command_status.eq("definite_no_test_command").all()


def test_identical_rows_are_retained_without_invented_adjudication():
    a = labels("definite_test_command")
    out = agreed_only(a, a.copy())
    assert out.test_command_status.eq("definite_test_command").all()
    assert out.unadjudicated_field_count.eq(0).all()


def test_constant_perfect_agreement_has_undefined_kappa():
    a = labels("definite_test_command")
    out = reliability(a, a.copy())
    assert out.sample_agreement.eq(1).all()
    assert out.cohen_kappa.isna().all()


def test_candidate_is_explicitly_automatic_and_final_fields_stay_empty():
    a, b = labels("definite_test_command"), labels("definite_no_test_command")
    candidate = labels("ambiguous")
    out = candidate_fields(a, b, candidate)
    assert len(out) == 2
    assert out.proposed_value.eq("ambiguous").all()
    assert out.final_value.eq("").all()
    assert out.final_source.eq("").all()
    assert not out.candidate_matches_a.any() and not out.candidate_matches_b.any()


def test_missing_window_is_rejected():
    a = labels("definite_test_command")
    with pytest.raises(ValueError, match="IDs/windows"):
        agreed_only(a, a.iloc[:1])
