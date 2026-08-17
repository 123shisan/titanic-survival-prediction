# -*- coding: utf-8 -*-
"""泰坦尼克 - 第三步：特征处理"""
import pandas as pd

df = pd.read_csv(r'C:\Users\86152\Desktop\机器学习\titanic\train.csv')

# 1. 标签编码：Sex 文字 -> 数字
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
# 2. 标签编码：Embarked 登船港 -> 数字
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

print('===== 编码后的数据（前5行）=====')
print(df.head())

print()
print('===== Sex列现在的值 =====')
print(df['Sex'].value_counts())

print()
print('===== Embarked列现在的值 =====')
print(df['Embarked'].value_counts())

# 3. 特征/标签分离：删掉 PassengerId 和 Survived
y = df['Survived']
X = df.drop(columns=['PassengerId', 'Survived'])

print()
print('===== 特征矩阵 X 形状 =====', X.shape)
print('X 的列:', list(X.columns))
print('===== 标签 y 形状 =====', y.shape)
