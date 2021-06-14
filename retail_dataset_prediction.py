
import pandas as pd 
import numpy as np
import streamlit as st
import warnings
warnings.filterwarnings("ignore")
st.set_option('deprecation.showPyplotGlobalUse', False)

df1 = pd.read_excel(r"C:\Users\Aswathy\Desktop\myenv\P52 - Retail Ecommerce\Retail Sector Analysis using RFM\no of customer per month.xlsx")
salepermonth = pd.read_csv(r"C:\Users\Aswathy\Desktop\myenv\P52 - Retail Ecommerce\Retail Sector Analysis using RFM\salepermonth.csv")
df = pd.read_csv(r"C:\Users\Aswathy\Desktop\myenv\P52 - Retail Ecommerce\Retail Sector Analysis using RFM\df_rfm.csv")
retail = pd.read_excel(r"C:\Users\Aswathy\Desktop\myenv\P52 - Retail Ecommerce\Sample_data.xlsx")
df2 = pd.read_csv(r"C:\Users\Aswathy\Desktop\myenv\P52 - Retail Ecommerce\ol111.csv")

st.title('Customer Likelihood Prediction Analysis')
st.write('Project 52 - Group 3')
nav = st.sidebar.radio('Pages',['Home','Prediction'])
if nav == 'Home':
    st.header("Data Set Details:")
    st.markdown(''' This is a sample dataset''')
    st.dataframe(retail)
    st.write(" Original Dataset contains {} number of Unique Customers.".format(df.shape[0]))
    
if nav == 'Prediction':
    st.title('Prediction')
    st.header(''' Prediction Status of Customers''')
    st.dataframe(df2)
    x = st.number_input("CustomerID: ",min_value=salepermonth['CustomerID'].min(),max_value=salepermonth['CustomerID'].max())
    z = pd.DataFrame(df.loc[df['CustomerID'] == x])
    y = pd.DataFrame(df2.loc[df2['CustomerID'] == x])
    if st.button("Predict"):
        if z.iloc[0,11] == 2:
            st.success('The Customer {} is likely to shop next month'.format(x))
            
            st.header("Likelihood Status")
            st.dataframe(y)
        else:
            st.warning('The Customer {} is not likely to shop next month'.format(x))
            st.header("Likelihood Status")
            st.dataframe(y)
            
        st.header("Money Spent per month by customer {}".format(x))
        p = salepermonth.loc[salepermonth['CustomerID'] == x]
        p = p.drop(p.iloc[:,0:1],axis = 1)
        fig = plt.figure(figsize = (20, 12))
        month = ['Dec-2010', 'Jan-2011', 'Feb-2011', 'Mar-2011', 'Apr-2011', 'May-2011', 'Jun-2011', 'Jul-2011',
              'Aug-2011', 'Sep-2011', 'Oct-2011', 'Nov-2011', 'Dec-2011']
        # creating the bar plot
        plt.plot(month,p.iloc[0] ,color ='blue')
        plt.xlabel("CustomerID: {}".format(x))
        plt.ylabel("Money Spent")
        st.pyplot()
        #st.write("Thank You")
