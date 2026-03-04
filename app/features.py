import pandas as pd
import numpy as np

def generate_features_for_prediction(df_hist):
    """
    输入：包含至少14天历史数据的DataFrame
         必须包含列: ['date', 'water_level', 'precipitation', 'temperature', 'humidity']
    输出：单行DataFrame，包含与训练时完全一致的35个特征
    """
    df = df_hist.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # ---------- 1. 基础降雨累积 ----------
    if 'precipitation' in df.columns:
        df['rain_3d'] = df['precipitation'].shift(1).rolling(3, min_periods=1).sum()
        df['rain_7d'] = df['precipitation'].shift(1).rolling(7, min_periods=1).sum()
        df['rain_14d'] = df['precipitation'].shift(1).rolling(14, min_periods=1).sum()

    # ---------- 2. 暴雨日与水位急涨 ----------
    if 'precipitation' in df.columns:
        df['heavy_rain_day'] = (df['precipitation'] > 50).astype(int)
        df['heavy_rain_days_3d'] = df['heavy_rain_day'].shift(1).rolling(3, min_periods=1).sum()
        df['heavy_rain_days_7d'] = df['heavy_rain_day'].shift(1).rolling(7, min_periods=1).sum()

    df['rapid_rise'] = (df['water_level'].shift(1).diff() > 0.5).astype(int)
    df['rapid_rise_days_3d'] = df['rapid_rise'].shift(1).rolling(3, min_periods=1).sum()
    df['rapid_rise_days_7d'] = df['rapid_rise'].shift(1).rolling(7, min_periods=1).sum()

    for window in [3, 7, 14]:
        df[f'water_level_mean_{window}d'] = df['water_level'].shift(1).rolling(window, min_periods=1).mean()
        df[f'water_level_max_{window}d'] = df['water_level'].shift(1).rolling(window, min_periods=1).max()

    for lag in [1, 2, 3, 7]:
        df[f'water_level_lag_{lag}d'] = df['water_level'].shift(lag)
        if 'precipitation' in df.columns:
            df[f'precip_lag_{lag}d'] = df['precipitation'].shift(lag)

    if 'precipitation' in df.columns:
        rain_gt0 = (df['precipitation'].shift(1) > 0).astype(int)
        df['rain_days_3d'] = rain_gt0.rolling(3, min_periods=1).sum()
        df['rain_days_7d'] = rain_gt0.rolling(7, min_periods=1).sum()

        def compute_api(series, decay=0.9):
            if len(series) == 0:
                return np.nan
            weights = decay ** np.arange(len(series))[::-1]
            return np.nansum(series * weights)
        df['api_7d'] = df['precipitation'].shift(1).rolling(7, min_periods=1).apply(compute_api, raw=False)

    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_rainy_season'] = ((df['month'] >= 5) & (df['month'] <= 7)).astype(int)

    if 'precipitation' in df.columns and 'water_level' in df.columns:
        df['precip_to_level_ratio'] = df['precipitation'] / (df['water_level'].shift(1) + 0.1)
        df['precip_to_level_ratio'] = df['precip_to_level_ratio'].clip(upper=10)

    if 'temperature' in df.columns and 'humidity' in df.columns:
        df['temp_humidity_index'] = df['temperature'] * (df['humidity'] / 100)

    if 'precipitation' in df.columns and 'humidity' in df.columns:
        df['wetness_index'] = (df['precipitation'] + 0.1) * (df['humidity'] / 100)

    # 删除NaN，只取最新一行
    df = df.dropna()
    return df.iloc[-1:]