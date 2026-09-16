import math
import numpy as np
import pytest
from research.beta_identification import (crown_drag, rotational_stiffness,
    free_element_rk4, identify, observations, run_document, BENCHMARK)


def analytic(t,beta=2,mass=5):
    # Independent exact solution to v'=g-(beta/m)v, position starts at [0,0].
    g=np.array([0.,-9.81]);v0=np.array([2.,1.]);k=beta/mass
    return np.array([g*x/k+(v0-g/k)*(1-math.exp(-k*x))/k for x in t])[:,None,:]


def forward(beta,t,dt):return free_element_rk4(t,beta,5,[0,0],[2,1],[0,-9.81],dt)


def test_drag_units_sign_and_crown_mass_partition():
    f=crown_drag(4,[1,3],[[2,0,0],[2,0,0]])
    np.testing.assert_allclose(f,[[-2,0,0],[-6,0,0]])
    assert np.sum(f[:,0])==-8
    assert rotational_stiffness(1e9,.1,1)==pytest.approx(1e9*math.pi*.1**4/64)
    with pytest.raises(ValueError):crown_drag(4,[0,0],[[1,0,0],[1,0,0]])


def test_rk4_converges_to_independent_solution():
    t=np.linspace(0,4,17);truth=analytic(t)
    errors=[np.max(abs(forward(2,t,dt)-truth)) for dt in [.25,.125,.0625]]
    assert errors[0]>errors[1]>errors[2]
    assert errors[0]/errors[1]>14


def test_recovers_known_beta_and_diagnostics_without_experimental_claim():
    t=np.linspace(0,3,10)
    result=identify(t,analytic(t),forward,bounds_kg_s=[0,6],bounds_reason='synthetic fixture bounds',
      forward_version=BENCHMARK,max_step_s=.05)
    assert result['beta_kg_s']==pytest.approx(2,abs=2e-5)
    assert result['numerically_converged']
    assert result['sensitivity_l2_m_per_kg_s']>0
    assert result['experimental_validation'] is False
    assert result['diagnostics']==[]


def test_flat_profile_does_not_report_an_identified_beta():
    t=[0,1,2];p=np.zeros((3,1,2))
    r=identify(t,p,lambda b,t,dt:p,bounds_kg_s=[0,4],bounds_reason='test',forward_version='constant',max_step_s=.1)
    assert r['beta_kg_s'] is None
    assert 'beta_not_identifiable_flat_profile' in r['diagnostics']


def test_boundary_and_invalid_or_nonfinite_data_are_explicit():
    t=np.linspace(0,2,7)
    r=identify(t,analytic(t,5),forward,bounds_kg_s=[0,3],bounds_reason='deliberately insufficient',forward_version=BENCHMARK,max_step_s=.05)
    assert 'optimum_at_search_boundary' in r['diagnostics']
    for times,p in [([0,0,1],np.zeros((3,1,2))),([0,1,2],np.full((3,1,2),np.nan))]:
        with pytest.raises(ValueError):observations(times,p)
    with pytest.raises(ValueError):run_document({'schema_version':1,'model':'real_tree','synthetic':False})


def test_input_version_sources_and_original_data_are_saved():
    t=[0,1,2]
    doc={'schema_version':1,'model':BENCHMARK,'synthetic':True,
      'units':{'length':'m','time':'s','mass':'kg','beta':'kg/s'},
      'times_s':t,'positions_m':analytic(t).tolist(),'max_step_s':.05,
      'bounds_kg_s':[0,6],'bounds_reason':'synthetic fixture',
      'parameters':{k:{'value':v,'source':'synthetic exact','standard_uncertainty':0}
        for k,v in {'mass_kg':5,'initial_position_m':[0,0],'initial_velocity_m_s':[2,1],'gravity_m_s2':[0,-9.81]}.items()}}
    result=run_document(doc)
    assert result['input']==doc and len(result['input_sha256'])==64
    assert result['scope']=='synthetic_benchmark_not_tree_identification'
    doc['units']['length']='cm'
    with pytest.raises(ValueError):run_document(doc)
