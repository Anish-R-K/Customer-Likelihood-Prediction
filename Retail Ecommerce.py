#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries

get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.metrics import classification_report,confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from __future__ import division
from sklearn.cluster import KMeans
import datetime

import chart_studio.plotly as py
import plotly.offline as pyoff
import plotly.graph_objs as go

import xgboost as xgb
from sklearn.model_selection import KFold, cross_val_score, train_test_split

import xgboost as xgb

#initate plotly
pyoff.init_notebook_mode()



#initiate visualization library for jupyter notebook 
#pyoff.init_notebook_mode()


# In[2]:


import pandas as pd
tx_data = pd.read_excel('Retail-Ecommerce.xlsx')


# In[3]:


tx_data.head(10)


# # Removing duplicate entries

# In[4]:


tx_data.drop_duplicates(inplace = True)


# # 1.Removing cancelled orders from the data

# In[5]:


invoices = tx_data['InvoiceNo']
x = invoices.str.contains('C', regex=True)
x.fillna(0, inplace=True)


# In[6]:


x = x.astype(int)
x = invoices.str.contains('C', regex=True)
x.fillna(0, inplace=True)
x = x.astype(int)
x.value_counts()
#A flag column was created to indicate whether the order corresponds to a canceled order.
tx_data['order_canceled'] = x
tx_data.head()


# In[7]:


tx_data['order_canceled'].value_counts()


# In[8]:


n1 = tx_data['order_canceled'].value_counts()[1]
n2 = tx_data.shape[0]
print('Number of orders canceled: {}/{} ({:.2f}%) '.format(n1, n2, n1/n2*100))


# In[9]:


tx_data = tx_data.loc[tx_data['order_canceled'] == 0,:]


# In[10]:


tx_data.reset_index(drop=True,inplace=True)


# # To remove CustomerID values which are missing and having negative quantity values

# In[11]:


tx_data = tx_data[tx_data['CustomerID'].notna()]
tx_data.reset_index(drop=True,inplace=True)


# In[12]:


tx_data


# In[13]:


tx_uk = tx_data[tx_data.Country == 'United Kingdom']


# In[14]:


tx_uk


# In[15]:


tx_uk.reset_index(drop=True,inplace=True)
tx_uk


# # New Customer Ratio

# # Customer Segmentation ::: Segmentation by RFM clustering

# RFM stands for Recency - Frequency - Monetary Value. Theoretically we will have segments like below:
#     
# Low Value: Customers who are less active than others, not very frequent buyer/visitor and generates very low - zero - maybe negative revenue.
# 
# Mid Value: In the middle of everything. Often using our platform (but not as much as our High Values), fairly frequent and generates moderate revenue.
# 
# High Value: The group we don’t want to lose. High Revenue, Frequency and low Inactivity.

# # Recency
# To calculate recency, we need to find out most recent purchase date of each customer and see how many days they are inactive for. After having no. of inactive days for each customer, we will apply K-means* clustering to assign customers a recency score.

# In[16]:


#convert the string date field to datetime
tx_data['InvoiceDate'] = pd.to_datetime(tx_data['InvoiceDate'])

#we will be using only UK data
tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)


# In[17]:


tx_data


# In[18]:


tx_uk


# Now we can calculate recency

# In[19]:


#create a generic user dataframe to keep CustomerID and new segmentation scores
tx_user = pd.DataFrame(tx_data['CustomerID'].unique())
tx_user.columns = ['CustomerID']

#get the max purchase date for each customer and create a dataframe with it
tx_max_purchase = tx_uk.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

#we take our observation point as the max invoice date in our dataset
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days

#merge this dataframe to our new user dataframe
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

tx_user.head()

#plot a recency histogram

plot_data = [
    go.Histogram(
        x=tx_user['Recency']
    )
]

plot_layout = go.Layout(
        title='Recency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[20]:


tx_user


# In[21]:


tx_user.Recency.describe()


# We see that even though the average is 90 day recency, median is 49.

# to apply K-means clustering to assign a recency score. But we should tell how many clusters we need to K-means algorithm. To find it out, we will apply Elbow Method. Elbow Method simply tells the optimal cluster number for optimal inertia. 

# In[22]:


from sklearn.cluster import KMeans

sse={}
tx_recency = tx_user[['Recency']]
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(tx_recency)
    tx_recency["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.show()


# Here it looks like 3 is the optimal one. Based on business requirements, we can go ahead with less or more clusters. We will be selecting 4 for this example:

# In[23]:


#build 4 clusters for recency and add it to dataframe
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

#function for ordering cluster numbers
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)
tx_user.groupby('RecencyCluster')['Recency'].describe()


# # Frequency
# To create frequency clusters, we need to find total number orders for each customer. First calculate this and see how frequency look like in our customer database:

# In[24]:


#get order counts for each user and create a dataframe with it
tx_frequency = tx_uk.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']
tx_frequency


# In[25]:


#add this data to our main dataframe
tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')
tx_user


# In[26]:



#plot the histogram
plot_data = [
    go.Histogram(
        x=tx_user.query('Frequency < 1000')['Frequency']
    )
]

plot_layout = go.Layout(
        title='Frequency'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[27]:


#k-means
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])
tx_user


# In[28]:



#order the frequency cluster
tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
tx_user


# In[29]:



#see details of each cluster
tx_user.groupby('FrequencyCluster')['Frequency'].describe()
tx_user


# As the same notation as recency clusters, high frequency number indicates better customers.

# # Revenue
# Let’s see how our customer database looks like when we cluster them based on revenue. We will calculate revenue for each customer, plot a histogram and apply the same clustering method.

# In[30]:


#calculate revenue for each customer
tx_uk['Revenue'] = tx_uk['UnitPrice'] * tx_uk['Quantity']
tx_revenue = tx_uk.groupby('CustomerID').Revenue.sum().reset_index()
tx_revenue


# In[31]:


#merge it with our main dataframe
tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')
tx_user


# In[32]:



#plot the histogram
plot_data = [
    go.Histogram(
        x=tx_user.query('Revenue < 10000')['Revenue']
    )
]

plot_layout = go.Layout(
        title='Monetary Value'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# In[33]:


#apply clustering
kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
tx_user


# In[34]:



#order the cluster numbers
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)
tx_user


# In[35]:



#show details of the dataframe
tx_user.groupby('RevenueCluster')['Revenue'].describe()


# # Overall Score
# We have scores (cluster numbers) for recency, frequency & revenue. Let’s create an overall score out of them:

# In[36]:


#calculate overall score and use mean() to see details
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user.groupby('OverallScore')['Recency','Frequency','Revenue'].mean()


# The scoring above clearly shows us that customers with score 8 is our best customers whereas 0 is the worst.
# To keep things simple, better we name these scores:
# 0 to 2: Low Value
# 3 to 4: Mid Value
# 5+: High Value

# In[37]:


tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value'


# In[38]:


tx_user


# In[39]:


#Revenue vs Frequency
tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Frequency'],
        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Frequency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

#Revenue Recency

tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Recency'],
        y=tx_graph.query("Segment == 'High-Value'")['Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Revenue"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)

# Revenue vs Frequency
tx_graph = tx_user.query("Revenue < 50000 and Frequency < 2000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Low-Value'")['Frequency'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['Recency'],
        y=tx_graph.query("Segment == 'Mid-Value'")['Frequency'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['Recency'],
        y=tx_graph.query("Segment == 'High-Value'")['Frequency'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "Frequency"},
        xaxis= {'title': "Recency"},
        title='Segments'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# The main strategies are quite clear:
# 
# High Value: Improve Retention
# 
# Mid Value: Improve Retention + Increase Frequency
# 
# Low Value: Increase Frequency

# # Customer Lifetime Value Prediction :- LTV prediction with XGBoost Multi-classification
#  Compnies invest in customers (acquisition costs, offline ads, promotions, discounts & etc.) to generate revenue and be profitable. Naturally, these actions make some customers super valuable in terms of lifetime value but there are always some customers who pull down the profitability. We need to identify these behavior patterns, segment customers and act accordingly.
#  
#  Lifetime Value: Total Gross Revenue - Total Cost
#  
# Define an appropriate time frame for Customer Lifetime Value calculation
# 
# Identify the features we are going to use to predict future and create them
# 
# Calculate lifetime value (LTV) for training the machine learning model
# 
# Build and run the machine learning model
# 
# Check if the model is useful

# # We will take 1 year of data, calculate RFM and use it for predicting next 1 months.

# In[40]:


tx_uk = tx_data.query("Country=='United Kingdom'").reset_index(drop=True)
tx_uk


# In[41]:


#create 1year and 1m dataframes
tx_1yr = tx_uk[(tx_uk.InvoiceDate < pd.to_datetime('2011-11-09')) & (tx_uk.InvoiceDate >= pd.to_datetime('2010-12-01'))].reset_index(drop=True)
tx_1m = tx_uk[(tx_uk.InvoiceDate > pd.to_datetime('2011-11-09')) & (tx_uk.InvoiceDate <= pd.to_datetime('2011-12-10'))].reset_index(drop=True)
tx_1yr


# In[42]:


tx_1m


# In[43]:


#create tx_user for assigning clustering
tx_user = pd.DataFrame(tx_1yr['CustomerID'].unique())
tx_user.columns = ['CustomerID']
tx_user


# In[44]:


tx_user.nunique()


# In[45]:



#order cluster method
def order_cluster(cluster_field_name, target_field_name,df,ascending):
    new_cluster_field_name = 'new_' + cluster_field_name
    df_new = df.groupby(cluster_field_name)[target_field_name].mean().reset_index()
    df_new = df_new.sort_values(by=target_field_name,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_field_name,'index']], on=cluster_field_name)
    df_final = df_final.drop([cluster_field_name],axis=1)
    df_final = df_final.rename(columns={"index":cluster_field_name})
    return df_final


# In[46]:


#calculate recency score
tx_max_purchase = tx_1yr.groupby('CustomerID').InvoiceDate.max().reset_index()
tx_max_purchase.columns = ['CustomerID','MaxPurchaseDate']
tx_max_purchase['Recency'] = (tx_max_purchase['MaxPurchaseDate'].max() - tx_max_purchase['MaxPurchaseDate']).dt.days
tx_user = pd.merge(tx_user, tx_max_purchase[['CustomerID','Recency']], on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Recency']])
tx_user['RecencyCluster'] = kmeans.predict(tx_user[['Recency']])

tx_user = order_cluster('RecencyCluster', 'Recency',tx_user,False)

tx_user


# In[47]:


#calcuate frequency score
tx_frequency = tx_1yr.groupby('CustomerID').InvoiceDate.count().reset_index()
tx_frequency.columns = ['CustomerID','Frequency']
tx_user = pd.merge(tx_user, tx_frequency, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Frequency']])
tx_user['FrequencyCluster'] = kmeans.predict(tx_user[['Frequency']])

tx_user = order_cluster('FrequencyCluster', 'Frequency',tx_user,True)
tx_user


# In[48]:


#calcuate revenue score
tx_1yr['Revenue'] = tx_1yr['UnitPrice'] * tx_1yr['Quantity']
tx_revenue = tx_1yr.groupby('CustomerID').Revenue.sum().reset_index()
tx_user = pd.merge(tx_user, tx_revenue, on='CustomerID')

kmeans = KMeans(n_clusters=4)
kmeans.fit(tx_user[['Revenue']])
tx_user['RevenueCluster'] = kmeans.predict(tx_user[['Revenue']])
tx_user = order_cluster('RevenueCluster', 'Revenue',tx_user,True)

tx_user


# In[49]:


#overall scoring
tx_user['OverallScore'] = tx_user['RecencyCluster'] + tx_user['FrequencyCluster'] + tx_user['RevenueCluster']
tx_user['Segment'] = 'Low-Value'
tx_user.loc[tx_user['OverallScore']>2,'Segment'] = 'Mid-Value' 
tx_user.loc[tx_user['OverallScore']>4,'Segment'] = 'High-Value' 


# In[50]:


tx_user


# There is no cost specified in the dataset. That’s why Revenue becomes our LTV directly.

# In[51]:


#calculate revenue and create a new dataframe for it
tx_1m['Revenue'] = tx_1m['UnitPrice'] * tx_1m['Quantity']
tx_user_1m = tx_1m.groupby('CustomerID')['Revenue'].sum().reset_index()
tx_user_1m.columns = ['CustomerID','m1_Revenue']


#plot LTV histogram
plot_data = [
    go.Histogram(
        x=tx_user_1m.query('m1_Revenue < 10000')['m1_Revenue']
    )
]

plot_layout = go.Layout(
        title='1m Revenue'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# Histogram clearly shows we have customers with negative LTV. We have some outliers too. 
# Filtering out the outliers makes sense to have a proper machine learning model.
# 
# next step. We will merge our 1yr and 1 months dataframes to see correlations between LTV and the feature set we have

# In[52]:


tx_1m


# In[53]:


tx_1yr.nunique()


# In[54]:


tx_1m.nunique()


# In[55]:


tx_merge = pd.merge(tx_user, tx_user_1m, on='CustomerID', how='left')
tx_merge = tx_merge.fillna(0)
tx_merge


# In[56]:


tx_graph = tx_merge.query("m1_Revenue < 30000")

plot_data = [
    go.Scatter(
        x=tx_graph.query("Segment == 'Low-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Low-Value'")['m1_Revenue'],
        mode='markers',
        name='Low',
        marker= dict(size= 7,
            line= dict(width=1),
            color= 'blue',
            opacity= 0.8
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'Mid-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'Mid-Value'")['m1_Revenue'],
        mode='markers',
        name='Mid',
        marker= dict(size= 9,
            line= dict(width=1),
            color= 'green',
            opacity= 0.5
           )
    ),
        go.Scatter(
        x=tx_graph.query("Segment == 'High-Value'")['OverallScore'],
        y=tx_graph.query("Segment == 'High-Value'")['m1_Revenue'],
        mode='markers',
        name='High',
        marker= dict(size= 11,
            line= dict(width=1),
            color= 'red',
            opacity= 0.9
           )
    ),
]

plot_layout = go.Layout(
        yaxis= {'title': "1m LTV"},
        xaxis= {'title': "RFM Score"},
        title='LTV'
    )
fig = go.Figure(data=plot_data, layout=plot_layout)
pyoff.iplot(fig)


# above merges our feature set and LTV data and plots LTV vs overall RFM score

# Considering business part of this analysis, we need to treat customers differently based on their predicted LTV. For this example, we will apply clustering and have 3 segments (number of segments really depends on your business dynamics and goals):
# Low LTV
# Mid LTV
# High LTV
# We are going to apply K-means clustering to decide segments and observe their characteristics:

# In[57]:


#remove outliers
tx_merge = tx_merge[tx_merge['m1_Revenue']<tx_merge['m1_Revenue'].quantile(0.99)]
tx_merge


# In[58]:



#creating 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(tx_merge[['m1_Revenue']])
tx_merge['LTVCluster'] = kmeans.predict(tx_merge[['m1_Revenue']])
tx_merge


# In[59]:


#order cluster number based on LTV
tx_merge = order_cluster('LTVCluster', 'm1_Revenue',tx_merge,True)
tx_merge


# In[60]:



#creatinga new cluster dataframe
tx_cluster = tx_merge.copy()

#see details of the clusters
tx_cluster.groupby('LTVCluster')['m1_Revenue'].describe()
tx_cluster


# There are few more step before training the machine learning model:
#     
# Need to do some feature engineering. We should convert categorical columns to numerical columns.
# 
# We will check the correlation of features against our label, LTV clusters.
# 
# We will split our feature set and label (LTV) as X and y. We use X to predict y.
# 
# Will create Training and Test dataset. Training set will be used for building the machine learning model. We will apply our model to Test set to see its real performance.
# 
# The code below does it all for us:

# In[61]:


#convert categorical columns to numerical
tx_class = pd.get_dummies(tx_cluster)
tx_class


# In[62]:


#calculate and show correlations
corr_matrix = tx_class.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)
corr_matrix


# In[63]:



#create X and y, X will be feature set and y is the label - LTV
X = tx_class.drop(['LTVCluster','m1_Revenue','Segment_High-Value','Segment_Low-Value','Segment_Mid-Value'],axis=1)
y = tx_class['LTVCluster']

#split training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=56)


# In[64]:


X


# In[65]:


tx_cluster


# In[66]:


X_test.shape


# Let’s start with the first line. get_dummies() method converts categorical columns to 0–1 notations. See what it exactly does with the example:

# This was our dataset before get_dummies(). We have one categorical column which is Segment. What happens after applying get_dummies():

# In[67]:


tx_class


# Segment column is gone but we have new numerical ones which represent it. We have converted it to 3 different columns with 0 and 1 and made it usable for our machine learning model.
# Lines related to correlation make us have the data below:
# 

# In[68]:


corr_matrix = tx_class.corr()
corr_matrix['LTVCluster'].sort_values(ascending=False)


# We see that 1 months Revenue, Frequency and RFM scores will be helpful for our machine learning models.
# Since we have the training and test sets we can build our model.

# In[69]:


#XGBoost Multiclassification Model
model=xgb.XGBClassifier(max_depth=5, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1)
ltv_xgb_model = model.fit(X_train, y_train)
print('Accuracy of XGB classifier on training set: {:.2f}'.format(ltv_xgb_model.score(X_train, y_train)))
print('Accuracy of XGB classifier on test set: {:.2f}'.format(ltv_xgb_model.score(X_test[X_train.columns], y_test)))
y_pred = ltv_xgb_model.predict(X_test)
print(classification_report(y_test, y_pred))


# In[70]:


X


# ## Predicting for the whole Dataset

# In[71]:


pred_all = ltv_xgb_model.predict(X)
pred_all


# In[72]:


CustomerID=X['CustomerID']
final_pred = pd.DataFrame(pred_all,columns=['Status'])
final_pred


# In[73]:


final_pred['Status'].replace(0,'No',inplace=True)
final_pred['Status'].replace(1,'No',inplace=True)
final_pred['Status'].replace(2,'Yes',inplace=True)
#final_pred


# In[74]:


X.to_csv('D:\Excelr academy\Projects\P52\Sub.csv',  index = False)


# In[75]:


# import the submission file which we have to submit on the solution checker.
submission = pd.read_csv('D:\Excelr academy\Projects\P52\Sub.csv')
submission


# In[76]:


submission['Status']=final_pred
submission.reset_index()
#converted the submission to .csv format.
pd.DataFrame(submission, columns=['CustomerID','Status']).to_csv('ol.csv',index=False)


# In[77]:


submission.head()


# In[78]:


import pickle
model=xgb.XGBClassifier(max_depth=5, learning_rate=0.1,objective= 'multi:softprob',n_jobs=-1)
pickle_out=open('Retail.pkl',mode='wb')
pickle.dump(model,pickle_out)
pickle_out.close()


# In[86]:


# df_rfm is downloaded from another file for deployment.
ol1 = pd.read_csv('df_rfm.csv')
ol1.head(10)


# In[88]:



ol1 = ol1.drop(['Recency','Frequency','Monetary','Time','r_quartile','f_quartile','m_quartile','t_quartile','RFM_score','RFM_Total'],axis=1)
ol1


# In[89]:



ol1['Cluster'].replace(0,'No',inplace=True)
ol1['Cluster'].replace(1,'No',inplace=True)
ol1['Cluster'].replace(2,'Yes',inplace=True)
ol1


# In[90]:





# In[ ]:




