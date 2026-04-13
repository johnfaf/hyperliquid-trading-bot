-- Feature store: OHLCV candle storage + computed features
-- Phase A of the quantitative infrastructure build

-- Raw OHLCV candles from Hyperliquid
CREATE TABLE IF NOT EXISTS candles (
    coin          TEXT             NOT NULL,
    timeframe     TEXT             NOT NULL,   -- '5m', '1h', '4h', '1d'
    timestamp_ms  BIGINT           NOT NULL,   -- epoch ms, candle open time
    open          DOUBLE PRECISION NOT NULL,
    high          DOUBLE PRECISION NOT NULL,
    low           DOUBLE PRECISION NOT NULL,
    close         DOUBLE PRECISION NOT NULL,
    volume        DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (coin, timeframe, timestamp_ms)
);

-- Primary lookup: latest N candles for a coin+timeframe
CREATE INDEX IF NOT EXISTS idx_candles_lookup
    ON candles (coin, timeframe, timestamp_ms DESC);

-- Computed features (EAV layout for flexibility)
CREATE TABLE IF NOT EXISTS features (
    coin          TEXT             NOT NULL,
    timeframe     TEXT             NOT NULL,
    timestamp_ms  BIGINT           NOT NULL,   -- aligned to candle open time
    feature_name  TEXT             NOT NULL,
    value         DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (coin, timeframe, timestamp_ms, feature_name)
);

-- ML consumer pattern: get feature X for coin Y over time
CREATE INDEX IF NOT EXISTS idx_features_lookup
    ON features (coin, timeframe, feature_name, timestamp_ms DESC);

-- Sync watermarks for incremental candle collection
CREATE TABLE IF NOT EXISTS candle_sync_state (
    coin              TEXT   NOT NULL,
    timeframe         TEXT   NOT NULL,
    last_timestamp_ms BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (coin, timeframe)
);
