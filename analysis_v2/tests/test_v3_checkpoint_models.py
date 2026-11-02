import hashlib
import numpy as np
import pandas as pd
import pytest
from p8audit.prefix_features import STRUCTURAL_FIELDS
from v3_checkpoint_models import prepare,score_cell


def data():
    rng=np.random.default_rng(8);n=120
    d=pd.DataFrame({f'k5_{field}':rng.normal(size=n) for field in STRUCTURAL_FIELDS})
    d['trajectory_id']=[f'SYNTHETIC-{i}' for i in range(n)]
    d['issue_id']=d.trajectory_id;d['repo']=[f'repo-{i//20}' for i in range(n)]
    d['scaffold']='sweagent';d['model']='synthetic-model'
    d['split']=np.repeat(['train','validation','test'],40)
    d['y']=np.tile([0,1],60)
    d['k5_reached']=1;d['k30_reached']=1
    d['k5_ran_tests']=rng.integers(0,2,n);d['k5_n_test_runs']=d.k5_ran_tests
    d['k5_test_fail']=rng.integers(0,2,n)*d.k5_ran_tests
    d['k5_fail_then_pass']=0
    return d


def test_prespecified_predictor_sets_and_log_count():
    d,fields=prepare(data(),5)
    assert len(fields['M0'])==len(STRUCTURAL_FIELDS)
    assert set(fields['M2'])-set(fields['M0'])=={'k5_ran_tests','k5_log_test_runs','k5_test_fail','k5_fail_then_pass'}
    np.testing.assert_allclose(d.k5_log_test_runs,np.log1p(d.k5_n_test_runs))


def test_score_cell_saves_then_reuses_predictions_and_checks_corruption(tmp_path):
    result=score_cell(data(),5,'main','logistic',tmp_path)
    path=tmp_path/'logistic_main_k5/predictions.parquet'
    original=path.read_bytes()
    again=score_cell(data(),5,'main','logistic',tmp_path)
    assert result==again
    assert path.read_bytes()==original
    assert result['models']['M2']['bootstrap_valid']==1000
    assert result['models']['M2']['delta_ap']==result['models']['M2']['ap']-result['models']['M0']['ap']
    path.write_bytes(b'corrupted synthetic artifact')
    with pytest.raises(ValueError,match='Saved predictions changed'):
        score_cell(data(),5,'main','logistic',tmp_path)
