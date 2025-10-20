create database finance;
psql -d finance -c "CREATE EXTENSION IF NOT EXISTS uuid-ossp;"



CREATE TABLE IF NOT EXISTS assets (
  id SERIAL PRIMARY KEY,
  symbol TEXT UNIQUE NOT NULL,               -- "SX5E", "EURUSD", "AAPL"
  asset_class TEXT NOT NULL CHECK (asset_class IN ('EQ','FX','INDEX','CRYPTO'))
);

CREATE TABLE IF NOT EXISTS prices (
  asset_id INT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL,
  close DOUBLE PRECISION NOT NULL CHECK (close > 0),
  PRIMARY KEY (asset_id, ts)
);

CREATE TABLE IF NOT EXISTS vol_surfaces (
  asset_id INT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
  asof DATE NOT NULL,
  tenor TEXT NOT NULL,                        -- "1M","3M","1Y"
  moneyness DOUBLE PRECISION NOT NULL,        -- K/S
  ivol DOUBLE PRECISION NOT NULL CHECK (ivol > 0),
  PRIMARY KEY (asset_id, asof, tenor, moneyness)
);

CREATE TABLE IF NOT EXISTS backtests (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),  -- si pgcrypto: DEFAULT gen_random_uuid()
  name TEXT NOT NULL,
  asset_id INT NOT NULL REFERENCES assets(id) ON DELETE RESTRICT,
  start_ts TIMESTAMPTZ NOT NULL,
  end_ts   TIMESTAMPTZ NOT NULL,
  params JSONB NOT NULL DEFAULT '{}',
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trades (
  backtest_id UUID NOT NULL REFERENCES backtests(id) ON DELETE CASCADE,
  ts TIMESTAMPTZ NOT NULL,
  position DOUBLE PRECISION NOT NULL,
  price DOUBLE PRECISION NOT NULL CHECK (price > 0),
  pnl   DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (backtest_id, ts)
);

CREATE TABLE IF NOT EXISTS metrics (
  backtest_id UUID NOT NULL REFERENCES backtests(id) ON DELETE CASCADE,
  metric TEXT NOT NULL,
  value DOUBLE PRECISION NOT NULL,
  PRIMARY KEY (backtest_id, metric)
);

CREATE INDEX IF NOT EXISTS idx_prices_asset_ts ON prices (asset_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_volsurf_asset_asof ON vol_surfaces (asset_id, asof);
CREATE INDEX IF NOT EXISTS idx_backtests_asset ON backtests (asset_id);
