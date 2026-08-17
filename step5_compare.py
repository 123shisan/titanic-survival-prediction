# -*- coding: utf-8 -*-
"""泰坦尼克 - 第五步：逻辑回归 vs 随机森林 对比"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ---- 读数据 + 特征处理（沿用之前）----
df = pd.read_csv(r'C:\Users\86152\Desktop\机器学习\titanic\train.csv')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
y = df['Survived']
X = df.drop(columns=['PassengerId', 'Survived'])

# ---- 拆训练/测试集 ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---- 模型1：逻辑回归 ----
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train, y_train)
acc_lr = accuracy_score(y_test, model_lr.predict(X_test))

# ---- 模型2：随机森林 ----
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)
acc_rf = accuracy_score(y_test, model_rf.predict(X_test))

# ---- 结果对比 ----
print('===== 模型对比 =====')
print('逻辑回归准确率: {:.1%}'.format(acc_lr))
print('随机森林准确率: {:.1%}'.format(acc_rf))
