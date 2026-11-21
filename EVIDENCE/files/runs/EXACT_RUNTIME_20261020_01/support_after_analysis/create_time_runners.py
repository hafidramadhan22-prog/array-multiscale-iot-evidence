"""Create isolated copies with the strongest existing port-free baseline."""
from pathlib import Path
import json,shutil
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    backup=ROOT/'history/20261020_before_TIME_RUNTIME';backup.mkdir(exist_ok=True)
    for rel in ['CURRENT_STATE.json','START_HERE.md','docs/PROTOCOL_TARGET.md','notebooks/TA_HAFID_TINJAUAN_HASIL.ipynb']:
        dest=backup/Path(rel).name
        if not dest.exists():shutil.copy2(ROOT/rel,dest)
    cfg=json.loads((ROOT/'configs/multiscale_repair.json').read_text());cfg.update(experiment='HOST_TIME_REPAIR_20261020',
        data_folder='runs/IAT_CONTEXT_DATA_20261020_01',source_run='runs/MULTISCALE_REPAIR_FULL_20261020_01',
        port_reference_run='runs/WB_RESEARCH_FULL_20261019_01',
        views=['time_context84','legacy25_time_context84','legacy25_age_context66'])
    out=ROOT/'configs/time_repair.json'
    if out.exists():raise ValueError('Configuration already exists')
    out.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
    for old,new in [('run_multiscale_repair.py','run_time_repair.py'),('verify_multiscale_repair.py','verify_time_repair.py'),
                    ('analyze_multiscale_repair.py','analyze_time_repair.py')]:
        text=(ROOT/'scripts'/old).read_text()
        if old.startswith('run_'):
            text=text.replace('from multiscale_context import repair_frame','from time_context import repair_frame')
            text=text.replace('scripts/run_multiscale_repair.py','scripts/run_time_repair.py').replace('scripts/verify_multiscale_repair.py','scripts/verify_time_repair.py')
            text=text.replace('src/multiscale_context.py','src/time_context.py').replace('tests/test_multiscale_context.py','tests/test_time_context.py')
            text=text.replace('configs/multiscale_repair.json','configs/time_repair.json').replace('docs/MULTISCALE_REPAIR_PROTOCOL_20261020.md','docs/TIME_RUNTIME_PROTOCOL_20261020.md')
            needle="    folder=ROOT/cfg['source_run']\n    items=json.loads((folder/'artifacts_sha256.json').read_text())"
            replacement="    folder=ROOT/cfg['source_run']\n    if 'legacy29_symmetric' in relative:folder=ROOT/cfg['port_reference_run']\n    relative=relative.replace('legacy25_symmetric','legacy25_multiscale60')\n    items=json.loads((folder/'artifacts_sha256.json').read_text())"
            if text.count(needle)!=1:raise ValueError('Baseline adapter location changed')
            text=text.replace(needle,replacement)
            text=text.replace("'requirements.lock.txt','src/context_repair.py'","'requirements.lock.txt','src/multiscale_context.py','docs/TIME_RUNTIME_LITERATURE_20261020.md','src/context_repair.py'")
        else:
            text=text.replace('from run_context_repair import','from run_time_repair import')
        if old.startswith('analyze_'):
            text=text.replace("'baseline25':'25 + konteks40 lama'","'baseline25':'25 + multiscale60 lama'")
            text=text.replace("'legacy25_multiscale_directional96':'25 + multiscale + arah'", "'legacy25_multiscale_directional96':'25 + multiscale + arah',\n'time_context84':'Konteks60 + waktu24','legacy25_time_context84':'25 + konteks60 + waktu24','legacy25_age_context66':'25 + konteks60 + age6'")
            text=text.replace('Multiscale60 memakai jendela0,1/1/10detik. Varian directional menambah36 statistik probe/respons berarah. Ring tetap1MiB; eviction kapasitas dan partial10s history dilaporkan. Bandingkan dua varian legacy25 untuk mengisolasi statistik arah.',
                'Baseline utama adalah WB25+multiscale60. Tambahan24fiturwaktu mengukur interarrival, recency dan observedspan. Kontrolage hanya mengambil6kolomrecency dari canonicalslot yang sama. Semua old60/arah36 direproduksi tepat; ring1MiB dan batascapture tetap.')
        dest=ROOT/'scripts'/new
        if dest.exists():raise ValueError('Tool exists: '+new)
        compile(text,new,'exec');dest.write_text(text,encoding='utf8')
    print('Created time-feature tools, configuration and history backup',flush=True)
