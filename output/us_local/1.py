python3 -c "
import json, sys
f = sys.argv[1]
with open(f) as fh: data = json.load(fh)
for t in ['AMD','AVGO','MU','NVDA','SMCI','MRVL']:
    d = data.get(t)
    if d:
        ms = d.get('model_score', 'N/A')
        cs = d.get('composite_score', 'N/A')
        ms_str = f'{ms:.4f}' if isinstance(ms, float) else str(ms)
        cs_str = f'{cs:.4f}' if isinstance(cs, float) else str(cs)
        print(f'{t}: rank={d.get(\"rank\",\"?\")} model={ms_str} composite={cs_str} bull_wl={d.get(\"in_bull_watchlist\",\"?\")}')
        sigs = d.get('signals') or d.get('composite_signals') or {}
        for k,v in sigs.items():
            print(f'   {k}: {v}')
    else:
        print(f'{t}: NOT FOUND')
" /mnt/data/artefacts/us_local/output/scores_detail_momentum_2023-12-07.json


python3 -c "
import json
with open('/mnt/data/artefacts/us_local/output/scores_detail_momentum_2023-12-07.json') as f:
    data = json.load(f)
# Print raw AMD record
import pprint
pprint.pprint(data.get('AMD'))
"
