import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("./ta_all_mini.csv")
df = df.iloc[:, 1:]
del df['Close']
del df['High']
del df['Low']
del df['volatility_bbh']
del df['volatility_atr']
del df['volume_nvi']
del df['volatility_bbl']
del df['volatility_bbhi']
del df['volatility_bbli']
del df['volatility_bbm']
del df['volatility_kcc']
del df['volatility_kch']
del df['volatility_kcl']
del df['volatility_dch']
del df['volatility_dcl']
del df['volume_em']
del df['trend_kst']
del df['trend_kst_diff']
del df['trend_ema_fast']
del df['trend_ema_slow']
del df['trend_ichimoku_a']
del df['trend_ichimoku_b']
del df['trend_visual_ichimoku_b']
del df['trend_visual_ichimoku_a']
#del df['momentum_kama']
del df['momentum_wr']
del df['momentum_stoch_signal']
del df['momentum_stoch']
del df['momentum_uo']
del df['momentum_rsi']
del df['trend_macd']
del df['trend_macd_signal']
del df['trend_adx_pos']
del df['trend_adx_neg']
del df['trend_vortex_diff']
del df['trend_cci']
del df['trend_kst_sig']
del df['trend_aroon_up']
del df['trend_aroon_ind']
del df['others_dlr']
del df['others_cr']
del df['others_dr']
df.to_csv("./ta_all_mini_clean.csv", index=False) 








