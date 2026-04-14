"""
Statistical validation framework for trading strategies.

Modules:
  - data_loader: Download OHLCV via ccxt and cache to parquet
  - strategy_adapter: Wrap any BaseStrategy for vectorized backtesting
  - walk_forward: Walk-Forward Analysis (IS/OOS rolling windows)
  - monte_carlo: Monte Carlo trade-sequence resampling
  - param_stability: Parameter grid stability / heatmap
"""
