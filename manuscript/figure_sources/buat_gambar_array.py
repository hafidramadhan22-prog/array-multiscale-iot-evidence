"""Buat sendiri lima gambar artikel Array dari CSV eksperimen tersimpan.

Jalankan: python buat_gambar_array.py
Satu gambar: python buat_gambar_array.py --figure 2
Folder baru: python buat_gambar_array.py --output hasil_edit_saya

Kode plotting ini disiapkan dengan bantuan AI. Peneliti perlu membaca,
menjalankan, memeriksa data/hasil, dan menyetujui gambar yang dipakai.
Tidak memakai image generator, API, koneksi jaringan, training atau model joblib.
Hasil PDF/SVG berupa vektor; PNG 300 dpi untuk preview/Word.
"""
from pathlib import Path
import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
OUTPUT = ROOT / 'hasil_gambar'
OUTPUT_FILES = []


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_inputs():
    """Cegah salah CSV: cocokkan hash enam input dengan manifest bawaan."""
    records = json.loads((ROOT / 'input_manifest.json').read_text(encoding='utf-8'))
    for record in records:
        path = ROOT / record['file']
        if not path.is_file():
            raise FileNotFoundError(f'Input tidak ditemukan: {path}. Ekstrak seluruh ZIP dahulu.')
        if sha256(path) != record['sha256']:
            raise ValueError(f'Input berubah: {path.name}. Periksa provenance sebelum menggambar.')
    return records


def save(fig, name):
    """Hanya menulis nama keluaran gambar di folder OUTPUT; CSV tidak diubah."""
    for suffix in ['pdf', 'svg', 'png']:
        path = OUTPUT / f'{name}.{suffix}'
        if suffix == 'png':
            fig.savefig(path, dpi=300)
        else:
            fig.savefig(path)
        OUTPUT_FILES.append(path)
    plt.close(fig)


def gambar_1():
    """Diagram alur penelitian. Posisi kotak/panah diatur eksplisit; tidak memakai image generator."""
    fig, ax = plt.subplots(figsize=(9.4, 6))
    ax.set(xlim=(0, 10), ylim=(0, 7))
    ax.axis('off')

    def box(x, y, w, h, text, color='#eaf1f6', fs=15):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', facecolor=color, edgecolor='#31546e', linewidth=1))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs)

    def arrow(a, b, label=None):
        ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'color': '#31546e', 'lw': 1.3})
        if label:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.1, label, fontsize=13, ha='center')
    box(0.3, 5.85, 9.4, 0.85, 'Edge development: prior observations retained\nA: experts; G: weights; C: order; V: reused evaluation')
    for x, text in [(0.3, 'Representation (RQ1)\nC40/C60 x O1/O10\nHistory ownership'), (3.6, 'Execution (RQ2)\nWV (f): label certificate\nShared preprocessing'), (6.9, 'Explanation (RQ3)\nFull WV (f) probabilities\nWindow/donor audits')]:
        box(x, 3.98, 2.8, 1.12, text, fs=14.5)
        arrow((x + 1.4, 5.76), (x + 1.4, 5.2))
    arrow((6.48, 4.53), (6.78, 4.53))
    box(0.3, 2, 4.35, 1.12, 'Observed alarms: payload tests\nGV (g), designed after observed errors\nWV execution/XAI do not transfer', '#fff1e7', 14.5)
    box(5.3, 2, 4.4, 1.12, 'Frozen external scoring: WV, RF, GV\nCIC: capture-scenario labels\nTON: mapped flow labels', '#e6f2ef', 14.5)
    arrow((1.7, 3.88), (1.7, 3.22))
    arrow((4.78, 2.55), (5.16, 2.55))
    arrow((5, 3.86), (7.1, 3.24))
    box(0.3, 0.2, 9.4, 0.9, 'Within-trace context gains; same-WV execution and explanation checks.\nExternal limits reported separately; no pooled five-class score.', fs=15)
    arrow((2.4, 1.9), (2.4, 1.2))
    arrow((7.5, 1.9), (7.5, 1.2))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.99)
    save(fig, 'revision_flow')


def gambar_2():
    """Ablasi C40/C60 dan O1/O10. Error bar adalah SD antar tiga fitting seed, bukan CI independen."""
    f = pd.read_csv(DATA / 'factorial_summary.csv')
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7))
    for ax, protocol in zip(axs, ['temporal', 'signature']):
        g = f.loc[(f.protocol == protocol) & (f.method == 'weighted_voting')].set_index('view').loc[['c40_o1', 'c40_o10', 'c60_o1', 'c60_o10']]
        vals = 100 * g.macro_f1
        ax.bar(range(4), vals, color=['#8798a4', '#aebac2', '#236b8e', '#50a6a0'])
        if protocol == 'temporal':
            ax.errorbar(range(4), vals, yerr=100 * g.macro_f1_sd, fmt='none', ecolor='black', capsize=3)
        for i, v in enumerate(vals):
            ax.text(i, v + (0.18 if protocol == 'temporal' else 0.7), f'{v:.3f}', ha='center', fontsize=10.7)
        ax.set_ylim(95 if protocol == 'temporal' else 65, 100)
        ax.set_xticks(range(4), ['C40/O1', 'C40/O10', 'C60/O1', 'C60/O10'], rotation=40, ha='right', fontsize=11)
        ax.set_ylabel('Macro-F1 (%)', fontsize=12)
        ax.set_title('Temporal mean' if protocol == 'temporal' else 'Signature diagnostic', fontsize=11.5)
    fig.tight_layout(pad=0.6)
    save(fig, 'revision_factorial')


def gambar_3():
    """Tiga puluh pasangan runtime sesi 3 November. Tidak dicampur dengan sesi 17 November."""
    rt = pd.read_csv(DATA / 'replay_raw.csv')
    pivot = rt.pivot(index='repeat', columns='mode', values='total_seconds')
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for mode, label, color in [('native_certified', 'Native certified', '#a46539'), ('compact_certified', 'Compact certified', '#216f94')]:
        axs[0].plot(pivot.index + 1, pivot[mode], '.-', lw=0.8, label=label, color=color)
    axs[0].set(xlabel='Pair', ylabel='Replay time (s)', ylim=(0.5, 2.9))
    axs[0].legend(fontsize=10.8, loc='upper right')
    delta = pivot.native_certified - pivot.compact_certified
    axs[1].bar(pivot.index + 1, delta, color=np.where(delta > 0, '#216f94', '#b35f4a'))
    axs[1].axhline(0, color='black', lw=0.8)
    axs[1].set(xlabel='Pair', ylabel='Native - compact (s)')
    fig.tight_layout(pad=0.6)
    save(fig, 'revision_runtime')


def gambar_4():
    """Bagian mean absolute attribution A32. Ini besar atribusi pada finite donor game, bukan efek kausal."""
    shares = pd.read_csv(DATA / 'xai_shares.csv')
    g = shares.query("background=='A32' and method=='weighted_voting'").set_index('group')
    values = [100 * g.loc[k, 'share'] for k in ['history_100ms', 'history_1000ms', 'history_10000ms']] + [100 * g.loc[~g.index.str.startswith('history'), 'share'].sum()]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.barh(['0.1 s history (20)', '1 s history (20)', '10 s history (20)', 'Six protocol groups'], values, color=['#78a8b8', '#3e8a9f', '#245b80', '#a6acaf'])
    ax.invert_yaxis()
    ax.set(xlabel='Mean absolute attribution share (%)', xlim=(0, 100))
    for i, v in enumerate(values):
        ax.text(v + 0.6, i, f'{v:.2f}%', va='center', fontsize=12)
    fig.tight_layout(pad=0.6)
    save(fig, 'revision_xai')


def gambar_5():
    """FPR setiap cohort/file. TON 1/2 memakai target Normal yang dipetakan melalui flow."""
    cached = pd.read_csv(DATA / 'metrics_all_cached.csv')
    later = pd.read_csv(DATA / 'metrics_later_blocks.csv')
    external = pd.read_csv(DATA / 'primary_policy_metrics_verified.csv')
    K = 'sql_guard_rf_rescue'
    left = []
    for policy in ['weighted_reference', K]:
        left.append([cached.loc[(cached.cohort == 'additional760') & (cached.policy == policy), 'FPR_percent'].iloc[0]] + [later.loc[(later.block == b) & (later.policy == policy), 'FPR_percent'].iloc[0] for b in [1, 2, 3]] + [cached.loc[(cached.cohort == 'V_seed42') & (cached.policy == policy), 'FPR_percent'].iloc[0]])
    right = [[external.loc[(external.source == source) & (external.policy == p), 'FPR_percent'].iloc[0] for source in ['CIC_Normal0', 'CIC_Normal3', 'TON_Ransom1', 'TON_Ransom2']] for p in ['weighted_reference', K]]
    fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.45))
    for ax, values, names, title, top in [(axs[0], left, ['Initial\nn=760', 'Block 1\nn=769', 'Block 2\nn=699', 'Block 3\nn=697', 'V Normal\nn=3,860'], '(a) Edge within-trace cohorts', 85), (axs[1], right, ['CIC N0', 'CIC N3', 'TON 1*', 'TON 2*'], '(b) External Normal targets', 55)]:
        xx = np.arange(len(names))
        for shift, ys, label, color in [(-0.18, values[0], 'Reference WV (f)', '#b66441'), (0.18, values[1], 'Guarded GV (g)', '#147d82')]:
            ax.bar(xx + shift, ys, 0.33, label=label, color=color)
            for x, y in zip(xx + shift, ys):
                ax.text(x, y + 0.6, f'{y:.2f}' if y < 2 else f'{y:.1f}', ha='center', fontsize=11)
        ax.set_xticks(xx, names, fontsize=11)
        ax.set_ylabel('FPR (%)', fontsize=12)
        ax.set_title(title, loc='left', fontsize=12)
        ax.set_ylim(0, top)
    axs[1].axhline(8, color='#a4293c', ls='--', lw=1, label='External target: 8%')
    axs[0].legend(fontsize=11, ncol=2, loc='upper right', frameon=False)
    axs[1].legend(fontsize=10.8, loc='upper right', frameon=False)
    fig.tight_layout(pad=0.6, h_pad=1.2)
    save(fig, 'revision_fpr_transfer')


def main():
    global DATA, OUTPUT
    parser = argparse.ArgumentParser(description='Gambar ilmiah Array dari CSV, tanpa training ulang.')
    parser.add_argument('--figure', choices=['all', '1', '2', '3', '4', '5'], default='all')
    parser.add_argument('--output', type=Path, default=ROOT / 'hasil_gambar')
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    # Output tidak boleh mengganti folder input/sumber.
    if OUTPUT == ROOT or OUTPUT == DATA or DATA in OUTPUT.parents:
        raise ValueError('Pilih folder output tersendiri, misalnya hasil_gambar.')
    records = check_inputs()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Gaya dasar. Untuk mengubah label/ukuran/warna, edit fungsi gambar_1--gambar_5.
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 12,
        'axes.spines.top': False, 'axes.spines.right': False,
        'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    selected = range(1, 6) if args.figure == 'all' else [int(args.figure)]
    for n in selected:
        globals()[f'gambar_{n}']()
        print(f'Figure {n} selesai.')
    run = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'script_sha256': sha256(Path(__file__)),
        'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__,
        'input_files': records,
        'outputs': [{'file': p.name, 'sha256': sha256(p)} for p in OUTPUT_FILES],
        'new_model_fits': 0, 'model_loading': 0, 'new_predictions': 0,
        'scope': 'Regeneration of stored scientific plots; Figure 1 is a hand-specified schematic.',
    }
    log = OUTPUT / f'run_figure_{args.figure}.json'
    log.write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(f'PDF/SVG/PNG tersimpan di: {OUTPUT}')
    print('Angka penelitian tidak diubah. Jalankan ulang di folder output berbeda bila ingin menyimpan versi lama.')


if __name__ == '__main__':
    main()
