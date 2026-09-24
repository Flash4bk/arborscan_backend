"""Summarize actual field CSV; never fills missing controls or uncertainty.
Usage: python tools/geometry_validation.py field.csv --output summary.json
Input columns: research/geometry_field_template.csv. One fixed-condition group per
object/experiment/app/algorithm/method/metric/unit/control/conditions.
"""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

KEYS=('object_id','experiment_id','app_version','algorithm_version','method','metric','unit','control_value','control_method','conditions')


def number(value):
    if value is None or not str(value).strip(): return None
    result=float(value)
    if not math.isfinite(result): raise ValueError('Non-finite field value')
    return result


def summarize(rows):
    groups=defaultdict(list); results=[]; seen=set()
    for index,row in enumerate(rows,2):
        if any(not row.get(k) for k in ('object_id','experiment_id','app_version','algorithm_version','method','metric','unit','repeat')):
            raise ValueError(f'Row {index}: missing identity/method/unit/repeat')
        if row['unit'] not in ('m','deg','1'): raise ValueError(f'Row {index}: use m, deg or 1')
        control=number(row.get('control_value')); value=number(row.get('result')); uncertainty=number(row.get('control_uncertainty'))
        if uncertainty is not None and uncertainty<0: raise ValueError('Negative uncertainty')
        if control is not None and not row.get('control_method'): raise ValueError('Control method required')
        key=tuple(row.get(k,'') for k in KEYS)
        identity=(key,row['repeat'])
        if identity in seen: raise ValueError('Duplicate repeat within condition group')
        seen.add(identity)
        error=value-control if value is not None and control is not None else None
        results.append({**row,'signed_error':error,'absolute_error':abs(error) if error is not None else None,
            'relative_error':error/abs(control) if error is not None and control!=0 else None,
            'control_uncertainty':uncertainty})
        groups[key].append((value,error))
    summaries=[]
    for key,rows in groups.items():
        values=[v for v,e in rows if v is not None]; errors=[e for v,e in rows if e is not None]
        summaries.append({**dict(zip(KEYS,key)), 'n_results':len(values),'n_paired':len(errors),
            'bias':statistics.mean(errors) if errors else None,
            'mae':statistics.mean(abs(e) for e in errors) if errors else None,
            'rmse':math.sqrt(statistics.mean(e*e for e in errors)) if errors else None,
            'repeatability_sample_sd':statistics.stdev(values) if len(values)>=2 else None})
    return {'observations':results,'groups':summaries,'interpretation':'Descriptive sample statistics, not validated accuracy or a confidence interval; review conditions and independent controls.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('csv');p.add_argument('--output',required=True);a=p.parse_args()
    if Path(a.csv).resolve()==Path(a.output).resolve(): p.error('Output must differ from input')
    with open(a.csv,encoding='utf-8-sig',newline='') as f: result=summarize(list(csv.DictReader(f)))
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
