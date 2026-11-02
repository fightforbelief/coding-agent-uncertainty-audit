from p8audit.prefix_features import STRUCTURAL_FIELDS, _features_at_checkpoint
from p8audit.trajectory_parser import parse_trajectory
from v3_checkpoint_features import CHECKPOINTS, features


def trajectory(n, trailing=False):
    result=[dict(role="user",content="synthetic issue",tool_calls=[])]
    for i in range(n):
        result.append(dict(role="assistant",content="run",tool_calls=[dict(id=f"t{i}",type="function",function=dict(name="bash",arguments='{"command":"pytest -q"}'))]))
        result.append(dict(role="tool",content="================ 1 passed in 0.01s ================",tool_calls=[]))
    if trailing:
        result.append(dict(role="assistant",content="unfinished",tool_calls=[dict(id="tail",type="function",function=dict(name="bash",arguments='{"command":"pytest -q"}'))]))
    return result


def test_prefix_only_structure_equals_frozen_full_trajectory():
    raw=trajectory(35)
    actual=features(raw,35,"sweagent")
    parsed=parse_trajectory(raw)
    for k in CHECKPOINTS:
        expected=_features_at_checkpoint(parsed,k,config="sweagent")
        assert all(actual[f"k{k}_{name}"]==expected[f"k{k}_{name}"] for name in STRUCTURAL_FIELDS)
        assert actual[f"k{k}_n_test_runs"]==k


def test_short_trajectory_keeps_visible_unfinished_command_but_not_as_completed():
    raw=trajectory(12,trailing=True)
    actual=features(raw,12,"sweagent")
    assert actual["k15_reached"]==0
    assert actual["k15_prefix_completed_rounds"]==12
    assert actual["k15_n_test_runs"]==13
    assert actual["k30_n_test_runs"]==13
