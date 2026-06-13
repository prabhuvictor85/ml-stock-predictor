"""Integration smoke test — run with: python test_integration.py"""
import sys, warnings
sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

print('=== Integration smoke test (synthetic data) ===')

# 1. Config
from pipeline.config import get_config
cfg = get_config('sp500')
print(f'1. Config OK: {cfg.market_id}')

# 2. Synthetic panel
np.random.seed(42)
n_days = 900
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
dates = pd.bdate_range('2022-01-03', periods=n_days)
rows = []
for t in tickers:
    price = 100.0
    for d in dates:
        price *= np.exp(np.random.normal(0.0003, 0.018))
        rows.append({
            'date': d, 'ticker': t,
            'open': price * 0.998, 'high': price * 1.012,
            'low': price * 0.988, 'close': price,
            'volume': np.random.randint(1_000_000, 5_000_000),
            'adv_20d_usd': price * 2_000_000,
            'market_cap_usd': price * 1e9,
            'sector': 'Technology',
            'in_universe': True,
            'group_date': d,
        })
panel = pd.DataFrame(rows).set_index(['date', 'ticker']).sort_index()
panel['group_date'] = (
    panel.index.get_level_values('date')
    .to_series()
    .dt.to_period('W')
    .apply(lambda p: p.end_time.normalize())
    .values
)
print(f'2. Panel: {len(panel)} rows, {panel.index.get_level_values("ticker").nunique()} tickers')

# 3. Benchmark
bm_close = pd.Series(
    100.0 * np.cumprod(np.exp(np.random.normal(0.0002, 0.012, n_days))),
    index=dates, name='benchmark_close',
)
print(f'3. Benchmark: {len(bm_close)} days')

# 4. Features
from pipeline.features.engineer import FeatureEngineer, FEATURE_PREFIX
fe = FeatureEngineer(cfg, bm_close)
panel = fe.build(panel)
feat_cols = [c for c in panel.columns if c.startswith(FEATURE_PREFIX)]
print(f'4. Feature engineering OK: {len(feat_cols)} features')

# 5. Targets
from pipeline.targets.builder import TargetBuilder
tb = TargetBuilder(cfg)
panel = tb.build(panel, bm_close)
print(f'5. Targets OK: cs_rank_20d non-null={panel["cs_rank_20d"].notna().sum()}')

# 6. CV splits
from pipeline.validation.cv import PurgedWalkForwardCV
cv = PurgedWalkForwardCV(n_folds=3, min_train_window=300)
folds = list(cv.split(panel))
print(f'6. CV splits: {len(folds)} folds generated')

# 7. LGBMRanker
fold_spec, train_idx, test_idx = folds[0]
train_p = panel.iloc[train_idx]
train_grp_p, train_groups = cv.build_group_array(train_p, min_group_size=3)
X_tr = train_grp_p[feat_cols].fillna(0)
y_tr = train_grp_p['cs_rank_20d'].fillna(0)
from pipeline.models.lgbm_ranker import LGBMRanker
ranker = LGBMRanker({'num_leaves': 31, 'learning_rate': 0.05, 'n_estimators': 50}, seed=42)
ranker.fit(X_tr, y_tr, train_groups)
print(f'7. LGBMRanker OK — best_iteration={ranker.model_.best_iteration}')

# 8. CatBoost
from pipeline.models.catboost_model import CatBoostModel
y_tr_cls = train_grp_p['top_quintile'].fillna(0).astype(int)
cb = CatBoostModel({'iterations': 50, 'learning_rate': 0.1}, seed=42)
cb.fit(X_tr, y_tr_cls)
print('8. CatBoost OK')

# 9. Calibrator
from pipeline.models.calibrator import ProbabilityCalibrator
probs = cb.predict_proba(X_tr)
cal = ProbabilityCalibrator()
cal.fit(probs, y_tr_cls.values)
cal_probs = cal.transform(probs)
ece = cal.expected_calibration_error(cal_probs, y_tr_cls.values)
print(f'9. Calibrator OK — ECE={ece:.4f}')

# 10. Ensemble
from pipeline.models.ensemble import EnsembleRanker
ens = EnsembleRanker(ranker, cb, cal)
test_p = panel.iloc[test_idx]
test_univ = test_p[test_p['in_universe'] == True].copy()
X_te = test_univ[feat_cols].fillna(0)
scores = ens.score(X_te)
print(f'10. Ensemble OK — scores min={scores.min():.3f} max={scores.max():.3f}')

# 11. Portfolio construction
from pipeline.portfolio.constructor import PortfolioConstructor
test_univ['group_date'] = test_univ.index.get_level_values('date').max()
score_series = pd.Series(scores, index=test_univ.index)
pc_ctor = PortfolioConstructor(cfg, top_n=3, weighting='equal')
_, weights = pc_ctor.construct(test_univ, score_series)
print(f'11. Portfolio OK — {len(weights)} holdings, weights sum={sum(weights.values()):.3f}')

# 12. Drift monitor
from pipeline.monitoring.drift_monitor import FeatureDriftMonitor
dm = FeatureDriftMonitor(cfg, feat_cols[:5])
dm.fit_baseline(train_grp_p[feat_cols[:5]])
latest_date = test_univ.index.get_level_values('date').max()
drift_df = dm.compute_weekly_drift(test_univ, latest_date)
print(f'12. Drift monitor OK — {len(drift_df)} features checked, alerts={drift_df["alert"].sum()}')

# 13. NDCG metric
from pipeline.validation.metrics import ndcg_at_k, precision_at_k
from pipeline.models.lgbm_ranker import cs_rank_to_label
test_univ2 = test_univ[test_univ['cs_rank_20d'].notna()].copy()
if len(test_univ2) >= 10:
    rel = cs_rank_to_label(test_univ2['cs_rank_20d']).values
    sc = scores[:len(test_univ2)]
    ndcg = ndcg_at_k(rel, sc, k=10)
    prec = precision_at_k(test_univ2['top_quintile'].fillna(0).astype(int).values, sc, k=10)
    print(f'13. Metrics OK — NDCG@10={ndcg:.4f}, Precision@10={prec:.4f}')
else:
    print('13. Metrics skipped (not enough non-NaN target rows in test)')

# 14. Performance reporter
from pipeline.backtest.reporter import PerformanceReporter
n = 52
gross = pd.Series(np.random.normal(0.002, 0.015, n), index=pd.date_range('2023-01-01', periods=n, freq='W'))
net = gross - 0.0003
bm = pd.Series(np.random.normal(0.001, 0.012, n), index=gross.index)
rpt = PerformanceReporter(gross, net, bm)
report = rpt.report()
print(f'14. PerformanceReporter OK — net_sharpe={report.net_sharpe:.3f}, max_dd={report.max_drawdown:.3f}')

print()
print('=== ALL 14 INTEGRATION CHECKS PASSED ===')

