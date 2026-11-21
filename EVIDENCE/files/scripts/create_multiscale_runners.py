"""Create isolated copies of reviewed finite experiment tools; never overwrite."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    cfg=json.loads((ROOT/'configs/context_repair.json').read_text())
    cfg.update(experiment='MULTISCALE_DIRECTIONAL_REPAIR_20261020',data_folder='runs/MULTISCALE_DATA_20261020_01',
        views=['multiscale60','legacy25_multiscale60','legacy25_multiscale_directional96'])
    target=ROOT/'configs/multiscale_repair.json'
    if target.exists():raise ValueError('Existing configuration')
    target.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
    pairs=[('run_context_repair.py','run_multiscale_repair.py'),('verify_context_repair.py','verify_multiscale_repair.py'),
           ('analyze_context_repair.py','analyze_multiscale_repair.py'),('profile_context_repair.py','profile_multiscale_repair.py')]
    for old,new in pairs:
        text=(ROOT/'scripts'/old).read_text()
        if old=='run_context_repair.py':
            text=text.replace('from context_repair import repair_frame,select_on_g','from multiscale_context import repair_frame\nfrom context_repair import select_on_g')
            text=text.replace('scripts/run_context_repair.py','scripts/run_multiscale_repair.py').replace('scripts/verify_context_repair.py','scripts/verify_multiscale_repair.py')
            text=text.replace('src/context_repair.py','src/multiscale_context.py').replace('configs/context_repair.json','configs/multiscale_repair.json')
            text=text.replace('docs/CONTEXT_REPAIR_PROTOCOL_20261020.md','docs/MULTISCALE_REPAIR_PROTOCOL_20261020.md').replace('tests/test_context_repair.py','tests/test_multiscale_context.py')
            text=text.replace("'requirements.lock.txt']","'requirements.lock.txt','src/context_repair.py','scripts/run_context_repair.py']")
        if old=='profile_context_repair.py':
            text=text.replace('from context_repair import repair_frame','from multiscale_context import repair_frame')
            text=text.replace('scripts/profile_context_repair.py','scripts/profile_multiscale_repair.py').replace('src/context_repair.py','src/multiscale_context.py')
        if old=='analyze_context_repair.py':
            text=text.replace("'legacy25_context64':'25 + konteks40 + rasio24'", "'legacy25_context64':'25 + konteks40 + rasio24',\n'multiscale60':'Konteks0,1/1/10detik', 'legacy25_multiscale60':'25 + multiscale60', 'legacy25_multiscale_directional96':'25 + multiscale + arah'")
            text=text.replace('Context40 menghapus semua fitur paket lama. Context64 menambah24 rasio deterministik tanpa state/informasi mentah baru. Legacy25_context64 mempertahankan25 fitur paket untuk mengisolasi efek penghapusan fitur tersebut.',
                'Multiscale60 memakai jendela0,1/1/10detik. Varian directional menambah36 statistik probe/respons berarah. Ring tetap1MiB; eviction kapasitas dan partial10s history dilaporkan. Bandingkan dua varian legacy25 untuk mengisolasi statistik arah.')
        dest=ROOT/'scripts'/new
        if dest.exists():raise ValueError('Existing tool: '+new)
        compile(text,new,'exec');dest.write_text(text,encoding='utf8')
    print('Created isolated phase-B tools and configuration')
