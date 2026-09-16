"""Independent SI identification tools; NOT the unpublished tree forward solver.

The only bundled forward model is an explicitly synthetic free-element benchmark.
See BETA_SPEC.md for equations, source gaps, and the experimental contract.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Callable
import numpy as np

METHOD_VERSION = 'beta-coordinate-sse-v1'
BENCHMARK = 'synthetic-free-element-rk4-v1'


def array(value, name):
    a = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(a)):
        raise ValueError(name + ' must be finite')
    return a


def positive(value, name, zero=False):
    if isinstance(value, bool): raise ValueError(name+' cannot be boolean')
    v = float(value)
    if not math.isfinite(v) or (v < 0 if zero else v <= 0):
        raise ValueError(name+' has invalid SI value')
    return v


def crown_drag(beta_kg_s, crown_masses_kg, relative_velocities_m_s):
    """Vector form of proof Eq.(3), opposite element-minus-air velocity."""
    beta = positive(beta_kg_s, 'beta', zero=True)
    masses = array(crown_masses_kg, 'crown masses')
    v = array(relative_velocities_m_s, 'velocity')
    if masses.ndim != 1 or np.any(masses < 0) or masses.sum() <= 0 or v.shape != (len(masses), 3):
        raise ValueError('Require nonnegative crown masses and one 3D velocity per element')
    return -beta * (masses / masses.sum())[:, None] * v


def rotational_stiffness(elastic_modulus_pa, diameter_m, length_m):
    e = positive(elastic_modulus_pa, 'E')
    d = positive(diameter_m, 'diameter')
    length = positive(length_m, 'length')
    return e * math.pi * d**4 / (64 * length)


def observations(times_s, positions_m):
    t, p = array(times_s,'times'), array(positions_m,'positions')
    if t.ndim != 1 or len(t)<3 or t[0] != 0 or np.any(np.diff(t)<=0):
        raise ValueError('At least three strictly increasing timestamps starting at zero required')
    if p.ndim != 3 or p.shape[0]!=len(t) or p.shape[1]<1 or p.shape[2] not in (2,3):
        raise ValueError('positions shape must be [time, element, xy or xyz] in metres')
    return t,p


def coordinate_objective(observed, predicted):
    a,b=array(observed,'observed'),array(predicted,'predicted')
    if a.shape!=b.shape or a.ndim!=3: raise ValueError('Coordinate shape mismatch')
    residual=a-b
    return float(np.sum(residual**2)),np.linalg.norm(residual,axis=2).mean(axis=1)


def free_element_rk4(times_s, beta_kg_s, mass_kg, initial_position_m,
                     initial_velocity_m_s, gravity_m_s2, max_step_s):
    """Synthetic benchmark only: Newton m v' = m g - beta v, no tree joints."""
    t = array(times_s,'times')
    if t.ndim!=1 or len(t)<2 or t[0]!=0 or np.any(np.diff(t)<=0): raise ValueError('Invalid time axis')
    beta=positive(beta_kg_s,'beta',zero=True); mass=positive(mass_kg,'mass');dt=positive(max_step_s,'step')
    r=array(initial_position_m,'initial position');v=array(initial_velocity_m_s,'initial velocity');g=array(gravity_m_s2,'gravity')
    if r.shape not in [(2,),(3,)] or r.shape!=v.shape or r.shape!=g.shape: raise ValueError('Invalid spatial dimensions')
    if sum(math.ceil(float(x)/dt) for x in np.diff(t))>1_000_000: raise ValueError('Integration budget exceeded')
    if beta*dt/mass>1: raise ValueError('Step too large for damping time; reduce max_step_s')
    y=np.concatenate([r,v]);n=len(r)
    def f(s): return np.concatenate([s[n:],g-beta/mass*s[n:]])
    out=[r.copy()]
    for delta in np.diff(t):
        steps=math.ceil(float(delta)/dt);h=delta/steps
        for _ in range(steps):
            k1=f(y);k2=f(y+h*k1/2);k3=f(y+h*k2/2);k4=f(y+h*k3)
            y=y+h*(k1+2*k2+2*k3+k4)/6
        if not np.all(np.isfinite(y)): raise ValueError('Integration diverged')
        out.append(y[:n].copy())
    return np.asarray(out)[:,None,:]


def identify(times_s, observed_m, forward: Callable, *, bounds_kg_s,
             bounds_reason, forward_version, max_step_s, beta_tolerance_kg_s=1e-5,
             trajectory_tolerance_m=1e-6, grid_size=25):
    """Adapter signature forward(beta, times, max_step) -> [time, element, coord].

    Profile-grid plus bounded golden refinement of the best grid neighbourhood.
    Does not assert global uniqueness or experimental validity.
    """
    t,observed=observations(times_s,observed_m)
    lo=positive(bounds_kg_s[0],'lower bound',zero=True);hi=positive(bounds_kg_s[1],'upper bound')
    dt=positive(max_step_s,'step');tol=positive(beta_tolerance_kg_s,'beta tolerance')
    pos_tol=positive(trajectory_tolerance_m,'trajectory tolerance')
    if hi<=lo or not bounds_reason or not forward_version or not 5<=grid_size<=1001:
        raise ValueError('Explicit finite bounds, justification, forward version and profile grid required')
    cache={}
    def evaluate(beta):
        if beta not in cache:
            predicted=forward(beta,t,dt)
            s,_=coordinate_objective(observed,predicted)
            if not math.isfinite(s): raise ValueError('Nonfinite objective')
            cache[beta]=s
        return cache[beta]
    grid=np.linspace(lo,hi,grid_size)
    scores=[evaluate(float(b)) for b in grid];best=int(np.argmin(scores))
    a=float(grid[max(best-1,0)]);b=float(grid[min(best+1,grid_size-1)])
    ratio=(math.sqrt(5)-1)/2
    for _ in range(150):
        if b-a<=tol:break
        c=b-ratio*(b-a);d=a+ratio*(b-a)
        if evaluate(c)<evaluate(d):b=d
        else:a=c
    evaluate((a+b)/2)
    beta=min(cache,key=cache.get)
    predicted=array(forward(beta,t,dt),'predicted')
    fine=array(forward(beta,t,dt/2),'refined')
    fine2=array(forward(beta,t,dt/4),'twice refined')
    s,mean=coordinate_objective(observed,predicted)
    step_delta=float(np.max(np.linalg.norm(predicted-fine,axis=2)))
    half_delta=float(np.max(np.linalg.norm(fine-fine2,axis=2)))
    step=max(tol,math.sqrt(np.finfo(float).eps)*max(1,beta))
    l=max(lo,beta-step);u=min(hi,beta+step)
    sensitivity=(array(forward(u,t,dt/4),'upper')-array(forward(l,t,dt/4),'lower'))/(u-l)
    flat=max(scores)-min(scores)<=np.finfo(float).eps*max(1,max(scores))*64
    flags=[]
    if flat:flags.append('beta_not_identifiable_flat_profile')
    if beta-lo<=tol or hi-beta<=tol:flags.append('optimum_at_search_boundary')
    if half_delta>pos_tol:flags.append('integration_tolerance_not_met')
    if b-a>tol:flags.append('optimizer_tolerance_not_met')
    return {'method_version':METHOD_VERSION,'forward_version':forward_version,
        'beta_kg_s':None if flat else beta,'candidate_beta_kg_s':beta,
        'sse_m2':s,'rmse_coordinate_m':float(np.sqrt(s/observed.size)),
        'times_s':t.tolist(),'observed_m':observed.tolist(),'predicted_m':predicted.tolist(),
        'residuals_m':(observed-predicted).tolist(),'mean_element_deviation_m':mean.tolist(),
        'profile':[{'beta_kg_s':float(x),'sse_m2':y} for x,y in zip(grid,scores)],
        'sensitivity_l2_m_per_kg_s':float(np.linalg.norm(sensitivity)),
        'integration_step_s':dt,'step_halving_max_delta_m':[step_delta,half_delta],
        'bounds_kg_s':[lo,hi],'bounds_reason':bounds_reason,'diagnostics':flags,
        'numerically_converged':not any('tolerance_not_met' in f for f in flags),
        'experimental_validation':False,'confidence_interval':None,
        'uniqueness_proven':False}


def run_document(document):
    if document.get('schema_version')!=1 or document.get('model')!=BENCHMARK or document.get('synthetic') is not True:
        raise ValueError('Only the explicitly synthetic benchmark is bundled; tree solver sources/data missing')
    if document.get('units')!={'length':'m','time':'s','mass':'kg','beta':'kg/s'}:
        raise ValueError('Explicit SI units required')
    p=document['parameters']
    for name in ['mass_kg','initial_position_m','initial_velocity_m_s','gravity_m_s2']:
        entry=p[name]
        if not entry.get('source') or 'standard_uncertainty' not in entry:
            raise ValueError('Parameter source and uncertainty (or null) required: '+name)
        if entry['standard_uncertainty'] is not None:
            if np.any(array(entry['standard_uncertainty'],'uncertainty')<0):raise ValueError('Negative uncertainty')
    def forward(beta,t,dt):
        return free_element_rk4(t,beta,*(p[k]['value'] for k in ['mass_kg','initial_position_m','initial_velocity_m_s','gravity_m_s2']),dt)
    result=identify(document['times_s'],document['positions_m'],forward,
        bounds_kg_s=document['bounds_kg_s'],bounds_reason=document['bounds_reason'],
        forward_version=BENCHMARK,max_step_s=document['max_step_s'])
    canonical=json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    result.update(input_sha256=hashlib.sha256(canonical).hexdigest(),input=document,
                  scope='synthetic_benchmark_not_tree_identification')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    if args.input.resolve()==args.output.resolve():parser.error('Output must not overwrite input')
    result=run_document(json.loads(args.input.read_text(encoding='utf-8')))
    with args.output.open('x',encoding='utf-8') as out:
        json.dump(result,out,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'scope':result['scope'],'beta_kg_s':result['beta_kg_s'],'diagnostics':result['diagnostics']}))
