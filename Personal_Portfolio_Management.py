#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[4]:


from pandas_datareader import data as wb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


stocks = ['BRITANNIA.NS', 'BHARTIARTL.NS', 'ULTRACEMCO.NS', 'DRREDDY.NS', 'M&M.NS', 'INFY.NS', 'HINDUNILVR.NS', 'GRASIM.NS', 'LT.NS', 'TECHM.NS']
data = pd.DataFrame()

for i in stocks:
    data[i] = wb.DataReader(i, data_source = 'yahoo', start = '2010-1-1', end = '2019-12-31')['Adj Close']


# In[6]:


data.tail()


# In[7]:


(data / data.iloc[0] * 100).plot(figsize=(10, 5))


# In[8]:


log_returns = np.log(data / data.shift(1))


# In[9]:


log_returns.cov()


# In[10]:


#weights = np.zeros(len(stocks)*simulations).reshape(simulations, len(stocks))
#weights

#weights[2, :] = np.random.random(len(stocks))
#weights[2, :], np.sum(weights[2, :])


# In[11]:


simulations = 50000
Expected_return = np.zeros(simulations)
Expected_vol = np.zeros(simulations)
weights = np.zeros(len(stocks)*simulations).reshape(simulations, len(stocks))


for i in range (simulations):
    
    weights[i, :] = np.random.random(len(stocks))
    weights[i, :] = weights[i, :]/np.sum(weights[i, :])
    
    Expected_return[i] = np.sum(weights[i, :] * log_returns.mean()) * 250
       
    Expected_vol[i] = np.sqrt(np.dot(weights[i, :].T,np.dot(log_returns.cov() * 250, weights[i, :])))
    
Expected_return, Expected_vol    


# In[12]:


portfolio_simul = pd.DataFrame({'Return-rate': Expected_return, 'Volatility':  Expected_vol})

for i in (stocks):
    portfolio_simul[i] = weights[:, stocks.index(i)]


# In[13]:


portfolio_simul.head()


# In[14]:


portfolio_simul.plot(x='Volatility', y='Return-rate', kind='scatter', figsize=(10, 6));
plt.xlabel('Volatility')
plt.ylabel('Expected Return')


# In[15]:


risk_free = 0.067


# In[16]:


sharpe = (Expected_return - risk_free)/Expected_vol


# In[17]:


np.argmin(sharpe)


# In[18]:


weights[np.argmin(sharpe),:]


# In[19]:


opt_pfol = portfolio_simul.loc[((portfolio_simul['Return-rate'] - risk_free)/portfolio_simul['Volatility']).idxmax()]


# In[21]:


plt.subplots(figsize = (20, 10))

plt.scatter(portfolio_simul['Volatility'], portfolio_simul['Return-rate'])
plt.scatter(opt_pfol[1], opt_pfol[0], color = 'g')
#plt.scatter(0.1, 0.1, color = 'b')
plt.plot([0, opt_pfol[1]], [risk_free, opt_pfol[0]], color = 'y')

