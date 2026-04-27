from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_RULE_COLUMNS = {'weight', 'target_label'}
OPTIONAL_SCORE_COLUMN = 'confidence'


def build_initial_rule_state(mln_rules_df: pd.DataFrame) -> dict:
    """Construct environment state from a mixed rule-pool DataFrame."""
    missing = REQUIRED_RULE_COLUMNS - set(mln_rules_df.columns)
    if missing:
        raise ValueError(f'Missing required rule columns: {sorted(missing)}')

    rule_pool = mln_rules_df.to_dict(orient='records')
    active_mask = np.ones(len(mln_rules_df), dtype=np.int64)
    weights = mln_rules_df['weight'].to_numpy(dtype=float)
    initial_weights = weights.copy()

    if OPTIONAL_SCORE_COLUMN in mln_rules_df.columns:
        rule_scores = mln_rules_df[OPTIONAL_SCORE_COLUMN].to_numpy(dtype=float)
    else:
        rule_scores = np.ones(len(mln_rules_df), dtype=float)

    target_labels = mln_rules_df['target_label'].to_numpy(dtype=np.int64)

    return {
        'rule_pool': rule_pool,
        'active_mask': active_mask,
        'weights': weights,
        'initial_weights': initial_weights,
        'rule_scores': rule_scores,
        'target_labels': target_labels,
        'num_rules': len(rule_pool),
    }
