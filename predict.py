# -*- coding: utf-8 -*-
"""泰坦尼克 - 第九步(2)：预测演示程序
用保存好的模型，预测一个新乘客是否存活。
运行方式：python predict.py
"""
import joblib
import pandas as pd

# ---- 加载保存好的模型（不用重新训练）----
model = joblib.load(r'C:\Users\86152\Desktop\机器学习\titanic\model.pkl')

# ---- 特征列名（必须和训练时一致）----
feature_names = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# ---- 造一个"新乘客"的特征（必须和训练时一样的顺序和编码）----
# 特征顺序: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
# Sex编码:   male=0, female=1
# Embarked: C=0, Q=1, S=2
passengers = [
    # 例子1：头等舱女性，38岁，票价81元  → 应该大概率存活
    [1, 1, 38, 0, 0, 81.76, 0],
    # 例子2：三等舱男性，25岁，票价0元  → 应该大概率遇难
    [3, 0, 25, 0, 0, 0.00, 2],
    # 例子3：三等舱小女孩，6岁 → 儿童，存活率高
    [3, 1, 6, 1, 2, 15.00, 1],
]

names = ['头等舱女性', '三等舱男性', '三等舱小女孩']
for name, p in zip(names, passengers):
    # 转成 DataFrame 并带上列名，消除特征名警告
    x = pd.DataFrame([p], columns=feature_names)
    pred = model.predict(x)[0]         # 0=死 1=活
    prob = model.predict_proba(x)[0]   # 各类别概率
    result = '存活' if pred == 1 else '遇难'
    print(f'{name}: 预测【{result}】 | 死亡概率{prob[0]:.2f}, 存活概率{prob[1]:.2f}')
