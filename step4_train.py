# -*- coding: utf-8 -*-
"""泰坦尼克 - 第四步：拆数据集 + 训练逻辑回归"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# ---- 读数据 ----
df = pd.read_csv(r'C:\Users\86152\Desktop\机器学习\titanic\train.csv')

# ---- 特征处理（沿用第三步）----
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Embarked'] = df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
y = df['Survived']
X = df.drop(columns=['PassengerId', 'Survived'])

# ---- 1. 拆训练集/测试集（80%训练 / 20%测试）----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print('训练集大小:', X_train.shape, '| 测试集大小:', X_test.shape)

# ---- 2. 创建逻辑回归模型 ----
model = LogisticRegression(max_iter=1000)

# ---- 3. 训练（让模型从训练集学规律）----
model.fit(X_train, y_train)

# ---- 4. 用测试集预测 ----
y_pred = model.predict(X_test)

# ---- 5. 评估：算准确率 ----
acc = accuracy_score(y_test, y_pred)
print()
print('===== 模型评估 =====')
print('测试集准确率: {:.1%}'.format(acc))
print()
print('预测的前10个结果(1=活 0=死):', y_pred[:10])
print('真实的前10个结果          :', y_test.values[:10])
