from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
from .model import load_model_components, predict_risk
from .features import generate_features_for_prediction

app = FastAPI(title="三都站洪水风险预测API")

# 允许所有来源跨域（部署后可限制为您的GitHub Pages域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 启动时加载模型（全局变量）
model, scaler, feature_cols, best_thresh = load_model_components()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    接收CSV文件，必须包含列: date, water_level, precipitation, temperature, humidity
    返回未来1天的风险预测结果
    """
    try:
        # 读取上传的CSV
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents), parse_dates=['date'])
        
        # 检查必要列
        required_cols = ['date', 'water_level', 'precipitation', 'temperature', 'humidity']
        if not all(col in df.columns for col in required_cols):
            raise HTTPException(status_code=400, detail=f"CSV必须包含列: {required_cols}")
        
        # 检查数据长度
        if len(df) < 14:
            raise HTTPException(status_code=400, detail="至少需要14天的历史数据")
        
        # 生成特征
        feature_row = generate_features_for_prediction(df)
        if feature_row.empty:
            raise HTTPException(status_code=400, detail="特征生成失败，请检查数据")
        
        # 预测
        risk_level, risk_label, high_prob = predict_risk(
            feature_row, model, scaler, feature_cols, best_thresh
        )
        
        # 计算预测日期（最后一天 + 1天）
        forecast_date = df['date'].max() + pd.Timedelta(days=1)
        
        return {
            "risk_level": risk_level,
            "risk_label": risk_label,
            "high_risk_probability": round(high_prob, 4),
            "forecast_date": str(forecast_date.date())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}