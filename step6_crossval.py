# -*- coding: utf-8 -*-
"""泰坦尼克 - 第六步：交叉验证科学评估"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# ---- 读数据 + 特征处理 ----
df = pd.read_csv(r'C:\Users\86152\Desktop\机器学习\titanic\train.csv')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
y = df['Survived']
X = df.drop(columns=['PassengerId', 'Survived'])

# ---- 逻辑回归：5折交叉验证 ----
lr = LogisticRegression(max_iter=1000)
scores_lr = cross_val_score(lr, X, y, cv=5)

# ---- 随机森林：5折交叉验证 ----
rf = RandomForestClassifier(n_estimators=100, random_state=42)
scores_rf = cross_val_score(rf, X, y, cv=5)

# ---- 结果 ----
print('===== 逻辑回归 5折交叉验证 =====')
print('每次的准确率:', scores_lr.round(3))
print('平均准确率  : {:.1%}'.format(scores_lr.mean()))

print()
print('===== 随机森林 5折交叉验证 =====')
print('每次的准确率:', scores_rf.round(3))
print('平均准确率  : {:.1%}'.format(scores_rf.mean()))
