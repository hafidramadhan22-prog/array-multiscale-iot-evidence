"""Rekomendasi holdout; angka operasional tetap keputusan desain penelitian."""
import numpy as np
import pandas as pd

METHODS = ('soft_voting', 'weighted_voting', 'stacking')


def rank_holdout(table, performance_weight=0.6,
                 protocol_status='ARAHAN_DOSPEM_HOLDOUT_2026_10_13_DESAIN_EKSPLISIT'):
    t = table.copy()
    expected = {f'{c}__{m}' for c in ('original', 'balanced') for m in METHODS}
    if len(t) != 6 or set(t.config) != expected or t.config.duplicated().any():
        raise ValueError('Ranking memerlukan tepat seluruh enam konfigurasi')
    if not np.isfinite(performance_weight) or not 0 <= performance_weight <= 1:
        raise ValueError('Parameter keputusan invalid')
    for row in t.itertuples():
        if row.config != f'{row.case}__{row.method}':
            raise ValueError('Identitas konfigurasi tidak konsisten')
    # Revisi pengguna 15 Oktober: semua enam berhak dibandingkan, tanpa filter treatment.
    t['eligible'] = True
    t['eligibility_reason'] = 'Seluruh enam dibandingkan tanpa prioritas Original/Balanced'
    t['balanced_minus_original_macro_f1'] = np.nan
    for method in METHODS:
        g = t[t.method == method]
        orig = g[g.case == 'original'].iloc[0]
        bal = g[g.case == 'balanced'].iloc[0]
        d = bal.macro_f1 - orig.macro_f1
        t.loc[g.index, 'balanced_minus_original_macro_f1'] = d
    cols = ['macro_f1', 'recall_macro', 'auc_ovr_macro', 'latency_batch_ms', 'delta_peak_mib']
    vals = t[cols].to_numpy(dtype=float)
    valid = np.isfinite(vals).all() and (vals[:, :3] >= 0).all() and (vals[:, :3] <= 1).all() and (vals[:, 3:] > 0).all()
    info = dict(protocol_status=protocol_status,
                evaluation_scope='comparative/exploratory historical holdout; not independent post-selection evaluation',
                normalization_scope='all_six', recommendation_policy='holdout_all_six',
                original_priority_applied=False,
                performance_weight=performance_weight,
                status='COMPLETE' if valid else 'INCOMPLETE_INVALID_METRICS',
                eligibility_complete=bool(t.eligible.notna().all()), winners=[])
    if not valid:
        t['tradeoff_score'] = np.nan
        t['rank_all_six'] = pd.NA
        info['reason'] = 'Perlu metrik lengkap serta latency dan delta peak positif; tanpa epsilon/fallback.'
        return t, t[t.eligible.fillna(False)].copy(), info
    t['performance_score'] = t[cols[:3]].mean(axis=1)
    t['latency_efficiency'] = t.latency_batch_ms.min() / t.latency_batch_ms
    t['memory_efficiency'] = t.delta_peak_mib.min() / t.delta_peak_mib
    t['efficiency_score'] = t[['latency_efficiency', 'memory_efficiency']].mean(axis=1)
    t['tradeoff_score'] = performance_weight*t.performance_score + (1-performance_weight)*t.efficiency_score
    t = t.sort_values(['tradeoff_score', 'config'], ascending=[False, True])
    def ranks(frame):
        result=[];reference=None;rank=0
        for position,score in enumerate(frame.tradeoff_score,1):
            if reference is None or not np.isclose(score,reference,rtol=0,atol=1e-12):
                reference=score;rank=position
            result.append(rank)
        return result
    t['rank_all_six'] = ranks(t)
    eligible = t[t.eligible].copy()
    eligible['rank_eligible'] = ranks(eligible)
    info['winners'] = eligible.loc[np.isclose(eligible.tradeoff_score, eligible.tradeoff_score.max(), rtol=0, atol=1e-12), 'config'].tolist()
    info['unrestricted_winners'] = t.loc[np.isclose(t.tradeoff_score, t.tradeoff_score.max(), rtol=0, atol=1e-12), 'config'].tolist()
    info['original_priority_changes_winner'] = set(info['winners']) != set(info['unrestricted_winners'])
    return t, eligible, info


def sensitivity_holdout(table, performance_weights=(0.5, 0.6, 0.7),
                         protocol_status='ARAHAN_DOSPEM_HOLDOUT_2026_10_13_DESAIN_EKSPLISIT'):
    """Analisis pendamping dari tabel yang sama; tidak mengubah aturan utama/fit."""
    import json
    summaries=[]; rankings=[]
    for weight in performance_weights:
        all_six,eligible,info=rank_holdout(table,weight,protocol_status)
        summaries.append(dict(performance_weight=weight,
            status=info['status'],winners=json.dumps(info['winners']),
            unrestricted_winners=json.dumps(info.get('unrestricted_winners',[])),
            recommendation_policy=info['recommendation_policy'],
            interpretation='descriptive sensitivity; not used to retune or choose main rule'))
        all_six['sensitivity_performance_weight']=weight
        rankings.append(all_six)
    return pd.DataFrame(summaries),pd.concat(rankings,ignore_index=True)
