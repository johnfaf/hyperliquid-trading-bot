## Continuous Learning And Decision Tightening

- Preserve source identity end to end so `polymarket`, `options_flow`, `arena_champion`, `copy_trade`, and native strategy signals train separately instead of collapsing into one generic bucket.
- Feed source-specific outcomes into AgentScorer, Kelly sizing, calibration, shadow attribution, and trade memory so the system compounds what is actually working over time.
- Update Alpha Arena training to score the specific winning or losing arena agent, not every agent with the same strategy type.
- Move Alpha Arena champion ideas into the main ranked decision funnel instead of executing them in a separate side path.
- Apply source-quality weights and calibration adjustments before the decision engine ranks candidates.
- Tighten decision gating with higher default score floors, higher confidence floors, fewer trades per cycle, and stricter external-signal thresholds.
- Keep Polymarket and options flow in the main decision stack while preserving their reasons, attribution keys, and confidence adjustments.
- Add regression coverage so external source keys, arena candidate injection, and arena-agent-specific learning stay locked in.
