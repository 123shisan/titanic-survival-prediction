# -*- coding: utf-8 -*-
"""泰坦尼克 - 第七步：特征工程（FamilySize）+ 对比"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r'C:\Users\86152\Desktop\机器学习\titanic\train.csv')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

y = df['Survived']
X_base = df.drop(columns=['PassengerId', 'Survived'])

# ---- 加新特征：家庭成员数 ----
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
X_new = df.drop(columns=['PassengerId', 'Survived'])

print('原来的特征:', list(X_base.columns))
print('新增特征后:', list(X_new.columns))
print()

# ---- 对比：加特征前 vs 加特征后（都用逻辑回归+5折交叉验证）----
lr = LogisticRegression(max_iter=1000)

s_base = cross_val_score(lr, X_base, y, cv=5).mean()
s_new = cross_val_score(lr, X_new, y, cv=5).mean()

print('===== 交叉验证对比 =====')
print('不加 FamilySize: {:.1%}'.format(s_base))
print('加   FamilySize: {:.1%}'.format(s_new))
