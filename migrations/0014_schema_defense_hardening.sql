-- Schema defense hardening: constraints, JSONB runtime metadata, and learning-table links.
-- Constraints are NOT VALID so existing legacy rows do not block deployment, while
-- new writes are protected immediately.

CREATE OR REPLACE FUNCTION _hl_try_jsonb(raw ANYELEMENT)
RETURNS JSONB
LANGUAGE plpgsql
IMMUTABLE
AS $$
BEGIN
    RETURN COALESCE(NULLIF(raw::text, '')::jsonb, '{}'::jsonb);
EXCEPTION WHEN others THEN
    RETURN '{}'::jsonb;
END;
$$;

ALTER TABLE IF EXISTS arena_agents
    ALTER COLUMN params TYPE JSONB USING _hl_try_jsonb(params),
    ALTER COLUMN params SET DEFAULT '{}'::jsonb;

ALTER TABLE IF EXISTS shadow_trades
    ALTER COLUMN metadata_json TYPE JSONB USING _hl_try_jsonb(metadata_json),
    ALTER COLUMN metadata_json SET DEFAULT '{}'::jsonb;

DROP FUNCTION IF EXISTS _hl_try_jsonb(ANYELEMENT);

ALTER TABLE IF EXISTS position_snapshots
    ADD CONSTRAINT chk_position_snapshots_positive_size CHECK (size > 0) NOT VALID,
    ADD CONSTRAINT chk_position_snapshots_positive_entry CHECK (entry_price > 0) NOT VALID,
    ADD CONSTRAINT chk_position_snapshots_positive_leverage CHECK (leverage IS NULL OR leverage > 0) NOT VALID,
    ADD CONSTRAINT chk_position_snapshots_side CHECK (LOWER(side) IN ('long', 'short', 'buy', 'sell')) NOT VALID;

ALTER TABLE IF EXISTS paper_trades
    ADD CONSTRAINT chk_paper_trades_positive_entry CHECK (entry_price > 0) NOT VALID,
    ADD CONSTRAINT chk_paper_trades_positive_exit CHECK (exit_price IS NULL OR exit_price > 0) NOT VALID,
    ADD CONSTRAINT chk_paper_trades_positive_size CHECK (size > 0) NOT VALID,
    ADD CONSTRAINT chk_paper_trades_positive_leverage CHECK (leverage IS NULL OR leverage > 0) NOT VALID,
    ADD CONSTRAINT chk_paper_trades_side CHECK (LOWER(side) IN ('long', 'short', 'buy', 'sell')) NOT VALID;

ALTER TABLE IF EXISTS wallet_fills
    ADD CONSTRAINT chk_wallet_fills_positive_original_price CHECK (original_price > 0) NOT VALID,
    ADD CONSTRAINT chk_wallet_fills_positive_penalised_price CHECK (penalised_price > 0) NOT VALID,
    ADD CONSTRAINT chk_wallet_fills_positive_size CHECK (size > 0) NOT VALID,
    ADD CONSTRAINT chk_wallet_fills_side CHECK (LOWER(side) IN ('long', 'short', 'buy', 'sell')) NOT VALID;

ALTER TABLE IF EXISTS shadow_trades
    ADD CONSTRAINT chk_shadow_trades_positive_entry CHECK (entry_price > 0) NOT VALID,
    ADD CONSTRAINT chk_shadow_trades_positive_exit CHECK (exit_price IS NULL OR exit_price > 0) NOT VALID,
    ADD CONSTRAINT chk_shadow_trades_positive_size CHECK (size > 0) NOT VALID,
    ADD CONSTRAINT chk_shadow_trades_side CHECK (LOWER(side) IN ('long', 'short', 'buy', 'sell')) NOT VALID;

ALTER TABLE IF EXISTS decision_outcomes
    ADD CONSTRAINT chk_decision_outcomes_label_win CHECK (label_win IS NULL OR label_win IN (-1, 0, 1)) NOT VALID,
    ADD CONSTRAINT chk_decision_outcomes_would_have_won CHECK (would_have_won IS NULL OR would_have_won IN (-1, 0, 1)) NOT VALID,
    ADD CONSTRAINT chk_decision_outcomes_side_correct CHECK (side_correct IS NULL OR side_correct IN (-1, 0, 1)) NOT VALID,
    ADD CONSTRAINT chk_decision_outcomes_side CHECK (side IS NULL OR LOWER(side) IN ('long', 'short', 'buy', 'sell')) NOT VALID;

ALTER TABLE IF EXISTS learning_backtest_runs
    ADD CONSTRAINT fk_learning_backtest_runs_dataset
    FOREIGN KEY (dataset_id) REFERENCES learning_datasets(dataset_id) ON DELETE SET NULL NOT VALID;

ALTER TABLE IF EXISTS learning_improvement_runs
    ADD CONSTRAINT fk_learning_improvement_runs_dataset
    FOREIGN KEY (dataset_id) REFERENCES learning_datasets(dataset_id) ON DELETE SET NULL NOT VALID;

ALTER TABLE IF EXISTS learning_policy_candidates
    ADD CONSTRAINT fk_learning_policy_candidates_parent
    FOREIGN KEY (parent_policy_id) REFERENCES continuous_learning_policies(policy_id) ON DELETE RESTRICT NOT VALID;

ALTER TABLE IF EXISTS learning_data_quality_reports
    ADD CONSTRAINT fk_learning_quality_dataset
    FOREIGN KEY (dataset_id) REFERENCES learning_datasets(dataset_id) ON DELETE CASCADE NOT VALID;

ALTER TABLE IF EXISTS learning_feature_attributions
    ADD CONSTRAINT fk_learning_attribution_dataset
    FOREIGN KEY (dataset_id) REFERENCES learning_datasets(dataset_id) ON DELETE CASCADE NOT VALID,
    ADD CONSTRAINT fk_learning_attribution_candidate
    FOREIGN KEY (candidate_policy_id) REFERENCES learning_policy_candidates(candidate_policy_id) ON DELETE SET NULL NOT VALID;

ALTER TABLE IF EXISTS learning_drift_reports
    ADD CONSTRAINT fk_learning_drift_baseline_dataset
    FOREIGN KEY (baseline_dataset_id) REFERENCES learning_datasets(dataset_id) ON DELETE CASCADE NOT VALID,
    ADD CONSTRAINT fk_learning_drift_current_dataset
    FOREIGN KEY (current_dataset_id) REFERENCES learning_datasets(dataset_id) ON DELETE CASCADE NOT VALID;

ALTER TABLE IF EXISTS learning_shadow_periods
    ADD CONSTRAINT fk_learning_shadow_periods_candidate
    FOREIGN KEY (candidate_policy_id) REFERENCES learning_policy_candidates(candidate_policy_id) ON DELETE CASCADE NOT VALID,
    ADD CONSTRAINT fk_learning_shadow_periods_champion
    FOREIGN KEY (champion_policy_id) REFERENCES continuous_learning_policies(policy_id) ON DELETE RESTRICT NOT VALID;

ALTER TABLE IF EXISTS learning_rollback_checks
    ADD CONSTRAINT fk_learning_rollback_candidate
    FOREIGN KEY (candidate_policy_id) REFERENCES learning_policy_candidates(candidate_policy_id) ON DELETE CASCADE NOT VALID,
    ADD CONSTRAINT fk_learning_rollback_policy
    FOREIGN KEY (rollback_policy_id) REFERENCES continuous_learning_policies(policy_id) ON DELETE RESTRICT NOT VALID;

ALTER TABLE IF EXISTS learning_promotion_packages
    ADD CONSTRAINT fk_learning_packages_candidate
    FOREIGN KEY (candidate_policy_id) REFERENCES learning_policy_candidates(candidate_policy_id) ON DELETE CASCADE NOT VALID,
    ADD CONSTRAINT fk_learning_packages_dataset
    FOREIGN KEY (dataset_id) REFERENCES learning_datasets(dataset_id) ON DELETE SET NULL NOT VALID,
    ADD CONSTRAINT fk_learning_packages_shadow_eval
    FOREIGN KEY (shadow_evaluation_id) REFERENCES learning_shadow_evaluations(evaluation_id) ON DELETE SET NULL NOT VALID,
    ADD CONSTRAINT fk_learning_packages_promotion_decision
    FOREIGN KEY (promotion_decision_id) REFERENCES learning_promotion_decisions(decision_id) ON DELETE SET NULL NOT VALID;

ALTER TABLE IF EXISTS learning_scheduler_runs
    ADD CONSTRAINT fk_learning_scheduler_dataset
    FOREIGN KEY (dataset_id) REFERENCES learning_datasets(dataset_id) ON DELETE SET NULL NOT VALID,
    ADD CONSTRAINT fk_learning_scheduler_improvement
    FOREIGN KEY (improvement_id) REFERENCES learning_improvement_runs(improvement_id) ON DELETE SET NULL NOT VALID,
    ADD CONSTRAINT fk_learning_scheduler_package
    FOREIGN KEY (package_id) REFERENCES learning_promotion_packages(package_id) ON DELETE SET NULL NOT VALID;

ALTER TABLE IF EXISTS learning_operator_reports
    ADD CONSTRAINT fk_learning_operator_reports_package
    FOREIGN KEY (package_id) REFERENCES learning_promotion_packages(package_id) ON DELETE SET NULL NOT VALID;