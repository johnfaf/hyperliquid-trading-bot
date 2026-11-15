ARG PILOT_IMAGE=hyperliquid-paper-pilot:28d8350f2f09
FROM ${PILOT_IMAGE}

COPY scripts/paper_exit_experiment.py /app/scripts/paper_exit_experiment.py
COPY src/backtest/paper_exit_study.py /app/src/backtest/paper_exit_study.py
COPY src/data/paper_exit_feed.py /app/src/data/paper_exit_feed.py

USER bot
HEALTHCHECK --interval=30s --timeout=8s --start-period=25s --retries=3 \
    CMD ["python", "-B", "scripts/paper_exit_experiment.py", "status", "--market-db", "/capture/btc-paper-exit-market.sqlite3"]
CMD ["python", "-B", "-u", "scripts/paper_exit_experiment.py", "capture", "--market-db", "/capture/btc-paper-exit-market.sqlite3", "--continuous"]
