# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
categorical data: fill NAN with 'nan'
"""

import numpy as np
import scipy as sp
import pandas as pd
import math
from scipy import stats 
import scipy.stats as ss
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pp

import numpy as np,gc
import matplotlib.dates as mdates
import scipy.stats
from scipy.stats import chi2

import datetime
import matplotlib.ticker as ticker

# =============================================================================
# # For dataset 
# =============================================================================
def summary_df(df):
    ## Basic info for each feature, incl. name, number of missing values, number of unique values, entropy 

    sum_tb = pd.DataFrame(df.dtypes,columns=['data_type'])
    sum_tb=sum_tb.reset_index()
    sum_tb['Feature'] = sum_tb['index']
    sum_tb=sum_tb[['Feature', 'data_type']]
    sum_tb['#_Missing']=df.isnull().sum().values
    sum_tb['%_Missing']=df.isnull().sum().values/df.shape[0]*100
    sum_tb['#_Unique'] = df.nunique().values
    sum_tb['%_Unique'] = df.nunique().values/df.shape[0]*100
    for fea in sum_tb['Feature'].value_counts().index:
        sum_tb.loc[sum_tb['Feature']==fea, 'Entropy'] = round(stats.entropy(df[fea].value_counts(normalize=True), base=2),2) 
    return sum_tb

def check_nan(df):
    print('Number of columns with missing value: ', df.isnull().any().sum(), '\n')
    for i in df.columns:
        print(i,'Number of NaN values: ',df[i].isnull().sum(),end=" ")
        print('Type:',df[i].dtypes,end=" ")
        print('% of NaN values:',np.round(df[i].isnull().sum()/df.shape[0]*100,2))

# =============================================================================
# # categorical 
# =============================================================================
def cross_frequency(df, col_1, col_2):
    cross_freq_tb=pd.crosstab(df[col_1].fillna('nan'), df[col_2].fillna('nan'), margins=True)
    
    # croos table (in percentage %)
    per_cross_freq_tb=pd.crosstab(df[col_1], df[col_2], normalize='index') 
    return cross_freq_tb, per_cross_freq_tb
    
def cat_cat_vis(df, col, label):
    df[col]=df[col].fillna('nan')
    val=df[label].unique()
    plt.figure(figsize=(8,6))
    g1 = sns.countplot(x=col, hue=label, data=df)
    plt.legend(title=label, loc='best', labels=list(val))
    g1.set_title('{} by {}'.format(col, label), fontsize=19)
    g1.set_xlabel(col, fontsize=17)
    g1.set_ylabel("Count", fontsize=17)
    
def cat_per_label_pivot(df, col, label, pivot):
    df[col]=df[col].fillna('nan')
    val=df[label].unique()
    
    temp = pd.crosstab(df[col], df[label], normalize='index') * 100
    temp=temp.reset_index()
   # temp.rename(columns=label_dict, inplace=True)
    
    plt.figure(figsize=(8,6))
    g1 = sns.countplot(x=col, hue=label, data=df)
    plt.legend(title=label, loc='best', labels=list(val))
    gt = g1.twinx()
    gt = sns.pointplot(x=col, y=pivot, data=temp, color='black', order=df[col].unique(), legend=False)
    g1.set_title('{} by {}'.format(col, label), fontsize=19)
    g1.set_xlabel(col, fontsize=17)
    g1.set_ylabel("Count", fontsize=17)
    
    
def chi_square(df, col_1, col_2, prob):# col_1: str
    contingency_table=pd.crosstab(df[col_1],df[col_2])
    Observed_Values = contingency_table.values
    stat, p, dof, expected =scipy.stats.chi2_contingency(contingency_table)
    

    print('=========', col_1,' & ', col_2, '=========')
    print('chi_square_statistic: ', stat)
    print('Degree of Freedom: ', dof)
    print('p_value: ', p)
    print('Probability: ', prob)
    
    critical = chi2.ppf(prob, dof)
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    
    '''
    #chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
    #chi_square_statistic=chi_square[0]+chi_square[1]
  #  critical_value=chi2.ppf(q=1-alpha,df=degree_free)
  #  critical_value=chi2.ppf(q=1-alpha,df=degree_free)
  #  p_value=1-chi2.cdf(x=chi_square_statistic,df=degree_free)

    print('Significance level: ',alpha)
    print('Degree of Freedom: ',degree_free)
    print('chi-square statistic:',chi_square_statistic)
    print('critical_value:',critical_value)
    print('p-value:',p_value)


    if chi_square_statistic>=critical_value:
        print(("Reject H0,There is a relationship between {} and {}").format(col_1, col_2))
    else:
        print(("Retain H0,There is no relationship between {} and {}").format(col_1, col_2))

    if p_value<=alpha:
        print(("Reject HA,There is a relationship between {} and {}").format(col_1, col_2))
    else:
        print(("Retain HA,There is no relationship between {} and {}").format(col_1, col_2))
    '''
def cramers_v(x, y):
    """
    correlation
    """
    confusion_matrix = pd.crosstab(x,y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
    rcorr = r-((r-1)**2)/(n-1)
    kcorr = k-((k-1)**2)/(n-1)
    return np.sqrt(phi2corr/min((kcorr-1),(rcorr-1)))
        
def cat_distribution(col, df):  # col_name: str
    plt.figure(figsize=(8,6))
    g = sns.countplot(x=col, data=df, order=df[col].unique())
    g.set_title("Distribution of "+col, fontsize=19)
    g.set_xlabel(col, fontsize=17)
    g.set_ylabel("Count", fontsize=17)
  #  g.set_ylim(0,500000)
    for p in g.patches:
        height = p.get_height()
        g.text(p.get_x()+p.get_width()/2., 
               height + 3,
               '{:1.2f}%'.format(height/len(df)*100),
               ha="center", fontsize=14)
        
def bi_cat_hist(df, col, label):
    color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
    color_idx = 0
  #  temp=[]
    for i in df[label].unique():
        tmp=df.loc[df[label]==i]
        
        plt.figure(figsize=(4, 2))
        tmp.groupby(col)[col].count().plot(kind='barh', title='Count of {} when {} = {}'.format(col, label, i), color=color_pal[color_idx])
        color_idx+=1
        plt.show()
        
def cat_continue_distribution(df, col, label):
    """
    categorical, but in numeric form
    e.g. df['card1'], df['card2'], df['card5']
    if df[col].dtype in ['float64','int64'] and len(df[col]>n)
    """
    color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
    color_idx = 0
    val=df[label].unique()
    for i in val:
        x=df[df[label]==i][col]
        plt.figure(figsize=(8,5))
        x.plot(kind='hist', title='{} when {}={}'.format(col, label, i), bins=50, color=color_pal[color_idx])
        color_idx+=1
        plt.show()
        
    plt.figure(figsize=(8,5))
    for i in val:
        plot_2 = sns.distplot(df[df[label] == i][col].dropna(), label='{}={}'.format(label, i), color=color_pal[color_idx])
        color_idx+=1
    plot_2.legend()
    plot_2.set_title('{} Values Distribution by {}'.format(col, label), fontsize=12)
    plot_2.set_xlabel('{} Values'.format(col), fontsize=10)
    plot_2.set_ylabel("Probability", fontsize=10)

        

def stacked_cat_cat_hist(df, col_1, col_2):
    # fill nan with -1
    df_plot = df.fillna('nan').groupby([col_1, col_2]).size().reset_index().pivot(columns=col_1, index=col_2, values=0).reset_index()
    g = df_plot.set_index(col_2).T.plot(stacked=True, color=sns.color_palette())
    sns.despine()
    g.figure.set_size_inches(8, 5)
    
    plt.show()
# =============================================================================
# # numeric
# =============================================================================
def num_quantiles(df, col_name):
    quan = df[col_name].quantile([.01, .025, .1, .25, .5, .75, .9, .975, 0.99])  # series
    return quan

def num_quantiles_by_label(df, col, label):
    val=df[label].unique()
    df_list=[]
    for i in val:
        df_list.append(df[df[label] == i][col].quantile([.01, .1, .25, .5, .75, .9, 1]).reset_index())
    quan_tg=pd.concat(df_list, axis=1, keys=val)
    quan_tg= quan_tg.rename(columns={'index': 'Quantile'})
    return quan_tg

def num_count_distribution_vis(df, col_name):
    if df[col_name].apply(np.log).isnull().values.any():
        plt.figure(figsize=(8, 5))
        df[col_name].plot(kind='hist',
              bins=100,
              title='Distribution of '+col_name)
    else:
        plt.figure(figsize=(10, 12))
        plt.subplot(211)
        df[col_name].plot(kind='hist',
              bins=100,
              title='Distribution of '+col_name)
        plt.subplot(212)
        df[col_name].apply(np.log).plot(kind='hist',
              bins=100,
              title='Distribution of log '+col_name)
    plt.show()     

def num_density_distribution_vis(df, col_name):
    plt.figure(figsize=(5,4))
    g=sns.distplot(df[col_name].dropna())
    g.set_title('Distribution of '+col_name, fontsize=18)
    g.set_xlabel("")
    g.set_ylabel("Probability", fontsize=15)

    
def num_distribution_by_label_vis(df, col, label):
    color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
    color_idx = 0
    val=df[label].unique()
    
    for i in val:
        plt.figure(figsize=(8,5))
        df.loc[df[label] == i][col].plot(kind='hist', bins=100,
          title='{} distribution when {} = {}'.format(col, label, i),
          color=color_pal[color_idx])
        color_idx+=1
        plt.show()
            
def num_combine_distribution(df, col, label):
    val=df[label].unique()
    plt.figure(figsize=(8,5))
    for i in val:
        plt1 = sns.distplot(df[df[label] == i][col].dropna(), label='{} = {}'.format(label, i))
        plt1.legend()
        plt1.set_title("{} distribution by {} ".format(col, label), fontsize=20)
        plt1.set_xlabel(col, fontsize=18)
        plt1.set_ylabel("Probability", fontsize=18)
    plt.show()
            
            
           
def corr_num_num(df, col_list):
    sns.set(font_scale=1.25)
    df=df[col_list]
    plt.figure(figsize=(20,20))
    sns.heatmap(df.corr(), cmap='RdBu_r', annot=True, center=0.0)
    plt.show()
    df_corr=df.corr()
    threshold=0.5
    corr_list=[]
    for i in range(0, len(col_list)):
        for j in range(i+1, len(col_list)):
            if (df_corr.iloc[i,j] >= threshold and df_corr.iloc[i,j] < 1) or (df_corr.iloc[i,j] < 0 and df_corr.iloc[i,j] <= -threshold):
                corr_list.append([df_corr.iloc[i,j],i,j]) 
    s_corr_list = sorted(corr_list,key=lambda x: -abs(x[0]))
    print('Highly correlated features: ')
    for v,i,j in s_corr_list:
        print("%s and %s = %.2f" % (col_list[i],col_list[j],v))
    return corr_list, s_corr_list
    
def num_pairplot(df, col_list, label):# binary 
    tmp = pd.concat([df.loc[df[label] == 0], df.loc[df[label] == 1]])
    sns.pairplot(tmp, hue=label, vars=col_list)
    plt.show()
    
def multi_pairplot(df, num_list, label):
    d=num_list
    d.append(label)
    sns.pairplot(df[d], hue=label)
    
    
def num_cat_label_inter(dff, cat, num, label):
    # cat: str, feature name, num: str, feature name, label: str
    # dff=df[df[num]<n]
    g = sns.boxenplot(x=cat, y=num, hue=label, data=dff)
    g.set_title(cat+' - '+num+' - '+label, fontsize=15)
    g.set_xlabel(cat, fontsize=10)
    g.set_ylabel(num, fontsize=10)
    plt.subplots_adjust(hspace = 0.6, top = 0.85)
    plt.show()  
    
def fill_with_others(df, col):
    # for categorical data with too many unique values
    dff=df.copy()
    level=dff[col].value_counts().describe()['50%']
    dff.loc[dff[col].isin(dff[col].value_counts()[dff[col].value_counts() < level].index), col] = "Others"
    return dff
    
# =============================================================================
# # time, plot 
# =============================================================================
def time_line_comntinue(df, date_col, col):
    # date_col shoulbe be in 
    data=df[[date_col, col]]
    
    data.set_index(date_col,inplace=True)
    fig, ax = plt.subplots(figsize=(15,7))
    data.plot(ax=ax)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(30))


def bilabel_count_time(df, date_col, label):
    """
    only for scinario with 2 labels
    individual y axis for each label
    """
    val=df[label].unique()
    
    temp = (df.groupby([date_col])[label]
                     .value_counts(normalize=False)
                     .rename('frequency')
                     .mul(100)
                     .reset_index()
                     .sort_values(date_col))
    temp_0=temp[temp[label]==val[0]]
    temp_1=temp[temp[label]==val[1]]
    fig = plt.figure(figsize=(15,7))
    ax1 = fig.add_subplot(111)
    ax1.plot(temp_0[date_col], temp_0['frequency'], color='blue')
    ax1.set_ylabel('{} = {}'.format(label, val[0]), color='blue')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(30))
    ax2 = ax1.twinx()
    ax2.plot(temp_1[date_col], temp_1['frequency'], color='red')
    ax2.set_ylabel('{} = {}'.format(label, val[1]), color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    
def multilabel_count_time(df, date_col, label):
    """
    if len(df[label])>2
    """
    color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]
    color_idx = 0
    
    val=df[label].unique()
    
    temp = (df.groupby([date_col])[label]
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values(date_col))
   
    
    fig = plt.figure(figsize=(15,7))
    for i in val:
        tmp=temp[temp[label]==i]
        plt.plot(date_col, 'percentage', data=tmp, color=color_pal[color_idx])
        plt.legend('{} = {}'.format(label, i))
        plt.ylabel('{} (perentage)'.format(label))
        color_idx+=1
   
    plt.show()
        

    

# =============================================================================
# # main     
# =============================================================================
def eda(df, cat_list, num_list, date_time, label):

    ######### overview of df #########
    print('Shape of dataframe: ', df.shape)
    print(df.info())
    
    # Nan value check
    print('Number of columns with missing value: ', df.isnull().any().sum(), '\n')
    
     ### summary table
    sum_tb=summary_df(df)
    print(sum_tb)
    
    ## Nan values check
    check_nan(df)
    
    ######### Categorical #########
    print(df[cat_list].info())
    ## uni-variate, categorical
    for cat in cat_list:
        # categorical distribution
        cat_distribution(cat, df)
        ## categorical versus label
        bi_cat_hist(df, cat, label)
        cat_cat_vis(df, cat, label)
        cat_per_label_pivot(df, cat, label, 1)
      #  stacked_cat_cat_hist(df, cat, label)
        
        if (len(df[cat].unique())>100) and (df[cat].dtype in ['float64','int64']):
            cat_continue_distribution(df, cat, label)
        
    ## bi-variate, categorical
    for col_1 in cat_list:
        for col_2 in sorted(cat_list, reverse=False):
            cross_freq_tb, per_cross_freq_tb = cross_frequency(df, col_1, col_2)
            print(cross_freq_tb)
            print(per_cross_freq_tb)
          #  stacked_cat_cat_hist(df, col_1, col_2)
            chi_square(df, col_1, col_2, 0.05)  # correlation, chi-square independent test
            #cramers_v(col_1, col_2)
        
        
    ######### Numeric #########  
    print(df[num_list].describe())  # overview 
    for num in num_list:
        quan = num_quantiles(df, num)
        quan_tg=num_quantiles_by_label(df, num, label)
        print('Quantiles of {}'.format(num))
        print(quan)
        print('Quantiles of {} by {}'.format(num, label))
        print(quan_tg)
        
        # uni-variate, distribution and logx distribution
        num_count_distribution_vis(df, num)
        
        ## numeric versus label
        num_distribution_by_label_vis(df, num, label)
        num_combine_distribution(df, num, label)
        time_line_comntinue(df, date_time, num)
        
    if len(df[label].unique())>2:
        multilabel_count_time(df, date_time, label)
    else:
        bilabel_count_time(df, date_time, label)
            
        
        
        
    
    ######### Correlation #########
    corr_num_num(df, num_list)
    num_pairplot(df, num_list, label)
        
    for cat in cat_list:
        for num in num_list:
            num_cat_label_inter(df, cat, num, label)  # multi-variate, 1 categorical + 1 numeric + label
            
    ######### Numeric data with time #########
    
   # time_line_comntinue(df, date_time, 'TransactionAmt')
    
    ######### Categorical data with time #########
    
    
    


#eda(df, ['card4', 'ProductCD'], ['TransactionAmt', 'dist1', 'dist2'], 'day', 'isFraud')

