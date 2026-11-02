import hashlib
import json

import pandas as pd
import pytest

from v3_5_ai_audit_report import STRATA, join_memberships, verify_lock, weighted_summary


def fixtures():
    groups = [("openhands", "minimax"), ("openhands", "qwen"), ("sweagent", "minimax"), ("sweagent", "qwen")]
    rows, cells = [], []
    for s in STRATA:
        for a, m in groups:
            N = 0 if s == STRATA[3] and (a, m) == groups[2] else 10
            if s == STRATA[2] and (a, m) == groups[0]:
                N = 30
            if s == STRATA[4] and (a, m) == groups[0]:
                N = 70
            cells.append(dict(stratum=s, scaffold=a, model=m, population_size=N, sample_size=int(N > 0), inclusion_probability=1/N if N else None))
            if not N:
                continue
            if s == STRATA[4] and (a, m) == groups[0]:
                row = next(r for r in rows if r["review_id"] == f"{STRATA[2]}-{a}-{m}")
            else:
                row = dict(review_id=f"{s}-{a}-{m}", issue_id=f"issue-{a}-{m}", scaffold=a, model=m)
                row.update({f"member_{t}": 0 for t in STRATA})
                row.update({f"pi_{t}": None for t in STRATA})
                rows.append(row)
            row[f"member_{s}"] = 1
            row[f"pi_{s}"] = 1/N
    key = pd.DataFrame(rows)
    labels = pd.DataFrame([dict(review_id=r["review_id"], window=w, test_command_status="definite_no_test_command") for r in rows for w in ("r5", "r15")])
    return labels, key, pd.DataFrame(cells)


def test_overlap_keeps_its_own_weights_and_drops_outcomes():
    labels, key, design = fixtures()
    key["resolved"] = 1
    data = join_memberships(labels, key, design)
    selected = data[(data.review_id == "C_r15_known_negative-openhands-minimax") & (data.window == "r15")]
    assert dict(zip(selected.stratum, selected.weight)) == {STRATA[2]: 30, STRATA[4]: 70}
    assert "resolved" not in data.columns
    assert len(data) == int(design.sample_size.sum()) * 2


def test_missing_or_duplicate_labels_fail():
    labels, key, design = fixtures()
    with pytest.raises(ValueError, match="Missing|both"):
        join_memberships(labels.iloc[1:], key, design)
    with pytest.raises(ValueError, match="Duplicate"):
        join_memberships(pd.concat([labels, labels.iloc[:1]]), key, design)


def test_wrong_pi_fails_instead_of_renormalising():
    labels, key, design = fixtures()
    key.loc[0, f"pi_{STRATA[0]}"] = .9
    with pytest.raises(ValueError, match="inclusion"):
        join_memberships(labels, key, design)


def test_empty_group_cannot_have_made_up_probability():
    labels, key, design = fixtures()
    design.loc[design.population_size == 0, "inclusion_probability"] = 1
    with pytest.raises(ValueError, match="Empty"):
        join_memberships(labels, key, design)


def test_weighted_abstention_bounds_are_not_confidence_intervals():
    frame = pd.DataFrame(dict(review_id=["a", "b", "c"], issue_id=["one", "one", "two"], weight=[2, 8, 10],
                              test_command_status=["definite_test_command", "definite_no_test_command", "ambiguous"]))
    result = weighted_summary(frame)
    assert result["positive_share_all"] == .1
    assert result["positive_share_determinate"] == .2
    assert result["uncertain_share"] == .5
    assert result["ambiguity_lower"] == .1
    assert result["ambiguity_upper"] == .6
    assert result["repeated_sample_issues"] == 1
    assert "confidence_interval" not in result


def test_all_abstentions_do_not_divide_by_zero():
    frame = pd.DataFrame(dict(review_id=["a"], issue_id=["one"], weight=[10], test_command_status=["ambiguous"]))
    result = weighted_summary(frame)
    assert result["positive_share_determinate"] is None
    assert result["ambiguity_upper"] == 1


def test_changed_locked_file_is_rejected(tmp_path):
    file = tmp_path / "annotations.csv"
    file.write_text("locked", encoding="utf8")
    lock = dict(files={file.name: hashlib.sha256(file.read_bytes()).hexdigest()}, label_lock_precedes_sampling_key_join=True)
    (tmp_path / "ANNOTATION_LOCK.json").write_text(json.dumps(lock), encoding="utf8")
    verify_lock(tmp_path)
    file.write_text("modified", encoding="utf8")
    with pytest.raises(ValueError, match="changed"):
        verify_lock(tmp_path)
