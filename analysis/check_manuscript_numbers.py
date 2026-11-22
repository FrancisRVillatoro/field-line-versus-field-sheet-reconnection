"""Check paper tables and key aggregate claims against preserved analysis data."""
from pathlib import Path
import argparse, json, re
import numpy as np
R=Path(__file__).resolve().parents[1]
a=json.loads((R/'data/scientific_validation.json').read_text())
b=json.loads((R/'data/small_amplitude_validation.json').read_text())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', nargs='*', type=Path, help='TeX source files containing the two numerical tables.')
args=parser.parse_args()
sources=args.source or [R/'data/reported_table_rows.tex']
text='\n'.join(source.read_text() for source in sources)
rows=a['observables']+b['observables']
def n(e,key): return next(r for r in rows if r['epsilon']==e and r['c']==.12 and r['observable']==key)
checks={}
for e in [.01,.02,.04,.08,.12]:
    row=next(l for l in text.splitlines() if l.startswith(f'{e:.2f} &') and len(l.split('&'))==5)
    fields=row.split('&')
    assert abs(float(fields[1])-n(e,'s_event')['observed'])<=5.1e-14
    assert abs(float(fields[2])-n(e,'paired_tau_event')['observed'])<=5.1e-14
    def parse_science(t):
        m=re.search(r'([\d.]+)\\times10\^{(-?\d+)}',t)
        assert m,t
        return float(m.group(1))*10**int(m.group(2))
    assert np.isclose(parse_science(fields[3]),n(e,'s_event')['residual'],rtol=5e-5,atol=0)
    assert np.isclose(parse_science(fields[4]),n(e,'s_event')['sensitivity_indicator'],rtol=.005,atol=0)
    checks[f'event_table_{e:.2f}']=True
assert len(rows)==45 and len(a['controls'])==45 and len(a['switches'])==18
assert all(v['switched'] for v in a['switches'])
assert min(v['minimum_gap_over_guard'] for v in a['switches'])>1.76
assert sum(v['diagnostic']['status']=='DIFFERENCES_UNRESOLVED' for v in a['refinement_diagnostics'])==52
assert max(v['sensitivity_indicator'] for v in a['observables'])<2.8e-11
assert min(v['residual_to_indicator'] for v in a['observables'])>1.04e6
assert b['minimum_residual_to_indicator']>1.13e4
assert max(v['max_V_norm'] for v in a['controls'] if v['max_V_norm'] is not None)<-.8600
assert min(v['min_lapse'] for v in a['controls'])>=.99919
tr=json.loads((R/'data/tracing_refinement.json').read_text())
assert sum(v['exits_fine'] for v in tr)==288
assert all(not v['classification_changes'] for v in tr)
assert max(v['max_endpoint_step_difference'] for v in tr)<4.75e-8
checks['aggregate_counts_sensitivities_causality_and_tracing']=True
for e in [.04,.08,.12]:
    row=next(l for l in text.splitlines() if l.startswith(f'{e:.2f} &') and len(l.split('&'))==4)
    v=next(v for v in a['controls'] if v['case']==f'e{round(e*100):02d}_N129_dt2')
    for cell,k in zip(row.split('&')[1:],['H','M','C']): assert np.isclose(parse_science(cell),v[k],rtol=.005,atol=0)
checks['constraint_table']=True
result={'status':'PASS','checks':checks,'scope':'Numerical table and aggregate consistency; no new evolution or statistical confidence claim.'}
(R/'data/manuscript_numbers_check.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
