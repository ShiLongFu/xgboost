import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('XGBoost_model.pkl')

# 定义特征的选项
SDH_options = {
    0: 'No (0)',
    1: 'Yes (1)'
}

tSAH_options = {
    0: 'No (0)',
    1: 'Yes (1)'
}

# Define feature names
feature_names = ["Hemoglobin", "Fibrinogen", "SDH", "Severe tSAH"]

# Streamlit的用户界面
st.title("Early Neurological Deterioration Predictor")

# Hemoglobin: 数值输入
Hemoglobin = st.number_input("Hemoglobin (g/L):", min_value=0, max_value=200, value=120)

# Fibrinogen: 数值输入
Fibrinogen = st.number_input("Fibrinogen (g/L):", min_value=0.01, max_value=20, value=1)

# contusion: 分类选择
SDH = st.selectbox("SDH:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# tSAH: 分类选择
Severe_tSAH = st.selectbox("Severe tSAH (Morris-Marshall grade 3 or 4):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# 处理输入并进行预测
feature_values = [Hemoglobin, Fibrinogen, SDH, Severe_tSAH]
features = np.array([feature_values])

if st.button("Predict"):
    # 预测类别和概率
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100

    if predicted_class == 1:
        advice = (f"According to our predictive model, this patient with PADBS has a high risk of early neurological deterioration. " f"The XGBoost model predicts that the probability of experiencing neurological deterioration is {probability:.1f}%.")
    else:
        advice = (f"According to our predictive model, this patient with PADBS has a low risk of early neurological deterioration. " f"The XGBoost model predicts that the probability of not experiencing neurological deterioration is {probability:.1f}%.")
        
    st.write(advice)

    # 计算SHAP值并显示力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)

    st.image("shap_force_plot.png")

# 运行Streamlit命令生成网页应用