# -*- coding: utf-8 -*-
"""泰坦尼克 - 第九步(1)：训练最终模型并保存"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib

# ---- 读数据 + 特征处理（最终版本）----
df = pd.read_csv(r'C:\Users\86152\Desktop\机器学习\titanic\train.csv')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
y = df['Survived']
X = df.drop(columns=['PassengerId', 'Survived'])

# ---- 训练最终模型 ----
model = LogisticRegression(max_iter=1000)
model.fit(X, y)   # 用全部数据训练（预测前不保留测试集，最大化利用数据）

# ---- 交叉验证评估（保留给项目README用）----
score = cross_val_score(model, X, y, cv=5).mean()
print('交叉验证准确率: {:.1%}'.format(score))

# ---- 保存模型到文件 ----
joblib.dump(model, r'C:\Users\86152\Desktop\机器学习\titanic\model.pkl')
print('模型已保存到: titanic\\model.pkl')
