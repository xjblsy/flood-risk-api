import joblib
import os
import pandas as pd
import numpy as np

# 获取当前文件所在目录的上级目录（项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

def load_model_components():
    """加载所有模型组件"""
    model = joblib.load(os.path.join(MODEL_DIR, 'xgboost_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    with open(os.path.join(MODEL_DIR, 'feature_cols.txt'), 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    with open(os.path.join(MODEL_DIR, 'best_threshold.txt'), 'r') as f:
        best_thresh = float(f.read().strip())
    return model, scaler, feature_cols, best_thresh

def predict_risk(feature_df, model, scaler, feature_cols, best_thresh):
    """
    feature_df: 已生成好的单行特征DataFrame
    返回: risk_level, risk_label, high_risk_probability
    """
    # 确保特征顺序与训练一致
    X_input = feature_df[feature_cols]
    X_scaled = scaler.transform(X_input)
    
    # 预测概率
    proba = model.predict_proba(X_scaled)[0]
    high_risk_prob = proba[2]
    
    # 原始多类预测
    y_pred = model.predict(X_scaled)[0]
    
    # 阈值优化
    if high_risk_prob >= best_thresh:
        risk_level = 2
    else:
        risk_level = y_pred
    
    risk_map = {0: '低风险', 1: '中风险', 2: '高风险'}
    return risk_level, risk_map[risk_level], high_risk_prob