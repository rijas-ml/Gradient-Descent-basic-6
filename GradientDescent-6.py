#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


# In[2]:


# x = Study hours (input feature)
# y = Corresponding exam scores (target variable)

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
y = np.array([35, 45, 50, 55, 65, 70, 75, 85, 90])


# In[5]:


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[6]:


model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.02, random_state=42)
model.fit(X_scaled, y)


# In[7]:


print("Slope (Weight):", model.coef_[0])
print("Intercept (Bias):", model.intercept_[0])


# In[8]:


y_pred = model.predict(X_scaled)


# In[9]:


for actual, pred in zip(y, y_pred):
    print(f"Actual: {actual:.2f} | Predicted: {pred:.2f}")


# In[10]:


plt.scatter(X, y, color='blue', label='Actual Scores')
plt.plot(X, y_pred, color='red', label='Predicted Line (GD)')
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Exam Score Prediction using Gradient Descent')
plt.legend()
plt.grid(True)
plt.show()


# In[11]:


jupyter nbconvert --to script "GradientDescent-6.ipynb"


# In[ ]:




