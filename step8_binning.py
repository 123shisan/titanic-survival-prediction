# -*- coding: utf-8 -*-
"""泰坦尼克 - 第八步：年龄分组（分箱）+ 对比"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

df = pd.read_csv(r'C:\Users\86152\Desktop\机器学习\titanic\train.csv')
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
y = df['Survived']

# ---- 版本1：用原始年龄 ----
X_age = df.drop(columns=['PassengerId', 'Survived'])

# ---- 版本2：把年龄分组 ----
df['Age'] = pd.cut(df['Age'], bins=[0, 12, 30, 50, 100], labels=[0, 1, 2, 3])
X_binned = df.drop(columns=['PassengerId', 'Survived'])

print('原始年龄的Age列前5个值:', X_age['Age'].head(5).tolist())
print('分组后的Age列前5个值:', X_binned['Age'].head(5).tolist())
print()

lr = LogisticRegression(max_iter=1000)
s_age = cross_val_score(lr, X_age, y, cv=5).mean()
s_bin = cross_val_score(lr, X_binned, y, cv=5).mean()

print('===== 交叉验证对比 =====')
print('用原始年龄   : {:.1%}'.format(s_age))
print('用分组年龄   : {:.1%}'.format(s_bin))
