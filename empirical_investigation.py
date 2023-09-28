#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Empirical assignment 2022 - 2023

@author: Max Temmerman and Marcio Reverbel

"""

import os
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
import pandas as pd

from helper import print_question, data_frame_to_latex_table_file

import math
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Start of Script for Empirical assignment Econometrics
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Set the folders for output of graphs and tables
# -----------------------------------------------------------------------------

# for the figures
figure_dir = '../figures/'
if not os.path.exists(figure_dir):
    os.makedirs(figure_dir)
# for the latex document
report_dir = '../report/'
if not os.path.exists(report_dir):
    os.makedirs(report_dir)

# -----------------------------------------------------------------------------
# Here we set the seed for our group to your group number
# -----------------------------------------------------------------------------

# first birthday
bd_1 = 706
# second birthday
bd_2 = 1106

group_seed = bd_1 * bd_2

# set the seed
np.random.seed(group_seed)

# -----------------------------------------------------------------------------
# set the random number generator and seed
# -----------------------------------------------------------------------------

# set the seed and the random number generator for reproducible results
rng = np.random.default_rng(group_seed)

# setting for output printing
print_line_length = 90
print_line_start = 5

# number of x points
num_points = 60

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
 
data_full = pd.read_csv('data.csv')

num_obs = 6000
# select 6000 observations randomly ( the rng uses your seed )
observations = rng.choice (len(data_full), num_obs , replace =False)
# select the observationsfor your group
data = data_full.iloc[ observations , :].copy()

# -----------------------------------------------------------------------------
# Descriptive statistics
# -----------------------------------------------------------------------------

print_question('Question 1: Descriptive Statistics')

# compute the summary statistics
data_summary = data.describe().round(2)

# print to screen
print(data_summary.T)

# export the summary statistics to a file
data_frame_to_latex_table_file(report_dir + 'summmary_stats.tex',
                               data_summary.T)

# -----------------------------------------------------------------------------
# Question 3
# -----------------------------------------------------------------------------

print_question('Question 3: Estimate first model')

# explanatory variables for question 3
x_vars_3 = data[['jc', 'univ','exper']]

# add a constant
X_3 = sm.add_constant(x_vars_3)

# get the dependent variable
y = data['lwage']

# set-up model 
model_3 = sm.OLS(y, X_3)

# estimate the model
results_3 = model_3.fit(use_t=False) 

# print the OLS output
print(results_3.summary())
print(results_3.summary2().tables[1])
# export the coefficients part of the summary to a table
data_frame_to_latex_table_file(report_dir + 'results_3.tex',
                               results_3.summary2().tables[1].round(4))

# -----------------------------------------------------------------------------
# Question 4
# -----------------------------------------------------------------------------

print_question('Question 4: Test JC vs UNIV')

# retrieve the covariance matrix of the parameters
sXX = results_3.cov_params()

# translate 
r_matrix = [0, 1, -1, 0]

# create ... test statistic

t_test = results_3.t_test(r_matrix, sXX)
print(t_test)

#t_list = [t_test.tvalue, t_test.pvalue]

# t_test.tvalue and -.pvalue returns arrays and not floats
# that's why here we hardcode the t-value and p-value into this tuple
t_tuple = (-1.2402696052926343, 0.21487569)

df_t_test = pd.Series(t_tuple, index=['t test','p-value']).round(4)

## exporting results t-test to latex table
data_frame_to_latex_table_file(report_dir + 't_test.tex',
                               np.round(df_t_test, 3))

                             
                               
# -----------------------------------------------------------------------------
# Question 5
# -----------------------------------------------------------------------------

print_question('Question 5: One-sided test of JC vs UNIV')

# explanatory variables for question 5
x_vars_5 = data[['jc', 'totcoll', 'exper']]

# add a constant
X_5 = sm.add_constant(x_vars_5)

# set-up model 
model_5 = sm.OLS(y, X_5)

# estimate the model
results_5 = model_5.fit(use_t=False) 

# print the OLS output
print(results_5.summary2().tables[1])

# export the coefficients part of the summary to a table
data_frame_to_latex_table_file(report_dir + 'results_5.tex',
                               results_5.summary2().tables[1].round(4))


# here I was thinking of manually doing the one-sided t test
t_1 = results_5.params['jc'] / results_5.bse['jc']
print(t_1)
alpha = 0.05
pval = stats.norm.sf(abs(t_1))
print(pval)
if pval < alpha:
    print('we reject H0')
else:
    print('we fail to reject H0')
# simply the same as dividing p-value from earlier two sided test by 2
stats.norm.cdf(t_1)
# for t-distribution: stats.t.sf(t_1, n-1)

t_oneside_tuple = (t_1, pval)

df_t_test_oneside = pd.Series(t_oneside_tuple, index=['t test','p-value']).round(4)

## exporting results one sided t-test to latex table
data_frame_to_latex_table_file(report_dir + 't_test_oneside.tex',
                               np.round(df_t_test_oneside, 3))

# -----------------------------------------------------------------------------
# Question 6
# -----------------------------------------------------------------------------

print_question('Question 6: Adding High-school rank')

# explanatory variables for question 6
x_vars_6 = data[['jc', 'univ', 'exper','phsrank']]

# add a constant
X_6 = sm.add_constant(x_vars_6)

# set-up model 
model_6 = sm.OLS(y, X_6)

# estimate the model
results_6 = model_6.fit(use_t=False) 

# print the OLS output
print(results_6.summary())

print(results_6.bse['phsrank'])


# export the coefficients part of the summary to a table
data_frame_to_latex_table_file(report_dir + 'results_6.tex',
                               results_6.summary2().tables[1].round(4))

b4 = results_6.params['phsrank']
print(math.exp(b4))
print( 100*( math.exp(b4) - 1 ), '% increase in wage for absolute increase of phsrank by 1' )
print( round(10* (100*( math.exp(b4) - 1 )),2), '% increase in wage for increase of phsrank by 10' )

# -----------------------------------------------------------------------------
# Question 8
# -----------------------------------------------------------------------------

print_question('Question 8: Model test')

# Using the same model (same explanatory variables) as defined in Question 6

## the wald test gives same results w/o specifiying covmatrix
covmatrix = results_6.cov_params()

R = np.eye(len(results_6.params))[1:5]
print(R)

model_test = results_6.wald_test(R, covmatrix)
print(model_test)
# the wald test gives same results w/o specifiying covmatrix

# Alternatively, the base fvalue given when fitting the OLS is for the null hypothesis 
# in which all parameters are equal to 0 (the model test), and also works here.
print(results_6.fvalue)

#print(np.eye(len(results_6.params))[1:5])

# export the coefficients part of the summary to a table
#data_frame_to_latex_table_file(report_dir + 'model_test',
                               #model_test.summary())
# -----------------------------------------------------------------------------
# Question 9
# -----------------------------------------------------------------------------

print_question('Question 9: Confidence and Prediction intervals')

beta = results_6.params
x = results_6.model.exog

ci = np.empty((2, num_obs))
for i in range(num_obs):
    ci[0,i] = x[i].T @ beta - 1.96 * results_6.mse_resid * np.sqrt(x[i].T @ la.inv(x.T @ x) @ x[i])
    ci[1,i] = x[i].T @ beta + 1.96 * results_6.mse_resid * np.sqrt(x[i].T @ la.inv(x.T @ x) @ x[i])
    
pi = np.empty((2, num_obs))
for i in range(num_obs):
    pi[0,i] = x[i].T @ beta - 1.96 * results_6.mse_resid * np.sqrt(1 + x[i].T @ la.inv(x.T @ x) @ x[i])
    pi[1,i] = x[i].T @ beta + 1.96 * results_6.mse_resid * np.sqrt(1 + x[i].T @ la.inv(x.T @ x) @ x[i])

y_hat = x @ beta

# Plotting the y vs y_hat graph, with confidence and prediction intervals.
fignum = 1

fig, ax = plt.subplots(1, 1, num=fignum)
ax.plot(y_hat, y, ',')
plt.plot(y_hat, ci[0], 'r--', lw=0.2)
plt.plot(y_hat, ci[1], 'r--', lw=0.2)
plt.plot(y_hat, pi[0], 'g--', lw=0.5)
plt.plot(y_hat, pi[1], 'g--', lw=0.5)
ax.grid(True, ls=':')
ax.set_xlabel('y_hat')
ax.set_ylabel('y')
ax.set_title("Confidence intervals and Prediction intervals")
ax.legend(loc='best')
plt.savefig(figure_dir + 'empirical_figure_{}.png'.format(fignum))
plt.show()

fignum += 1

# -----------------------------------------------------------------------------
# Question 10
# -----------------------------------------------------------------------------

print_question('Question 10: Testing for Heteroskedasticity')

white_test = sm.stats.diagnostic.het_white(results_6.resid, X_6)
# also possible to use results_6.model.exog instead of X_6
print(white_test)
print('F-value: {}, with p-value of {}'.format(white_test[2], white_test[3]))

df_f_test = pd.Series(white_test[2:], index=['F test','p-value']).round(4)

## exporting results white test to latex table
data_frame_to_latex_table_file(report_dir + 'white_test.tex',
                               np.round(df_f_test, 2))

## code and export to latex table adapted from White test slide in wpo9-10 answer video

### if needed adjust the std errors to heteroskedasticity consistent errors :
hc_se = results_6.HC0_se
print(hc_se)
hc_cov = results_6.cov_HC0
hc_model_test = results_6.wald_test(R, hc_cov) 
hc_F_test = results_6.f_test(R, hc_cov)
print(hc_model_test)
print(hc_F_test)

# -----------------------------------------------------------------------------
# Question 11
# -----------------------------------------------------------------------------

print_question('Question 11: Improving the model')
# 5 groups of variables
# remove exper and phsrank:
current = ['jc', 'univ', 'exper']
alternative = ['BA', 'AA', 'exper']
group = [['female'],['black', 'hispanic'],['stotal'],['smcity', 'medcity', 'submed', 'lgcity', 'vlgcity', 'subvlg'],
         ['ne', 'nc', 'south']]
# Bachelors and Associates degrees are particular cases because they very likely 
# measure the very similar things as 'jc' and 'univ'. The difference is that one pair is
# binary, while the other is continuous. Otherwise, we might stipulate that they
# would be collinear (or measure an added effect when graduating). 

for i in range(len(group)):
    new_data = data[current + group[i]]
    new_X = sm.add_constant(new_data)
    new_model = sm.OLS(y, new_X)
    new_results = new_model.fit()
    #print(new_results.summary())
    print(new_results.summary2().tables[1].round(4))

# based on this, we see the female, black, stotal, lgcity, vlgcity, nc and south all have statistically and economically significant impact on the model.

"""Change here from 'current' to 'alternative' to see the difference. Include all 
and you one can see that the presence of all these variables diminishes their effect
in the model. In essence, they are measuring the very similar things. In the end, we
opt for 'jc' and 'univ' because, since they are continuous variables, they offer
more nuance to the model. Addidionally, the existance of the totcoll variable, 
as previously demonstrated, allows us to make one-sided tests when comparing these
'jc' and 'univ'. That is not possible with the AA and BA variables. Additionally, the
(adjusted) R-squared value is higher when using 'jc' and 'univ'. An argument could be
made for the inclusion of all four variables, in such a way that 'jc' and 'univ' measures the 
effects of studies, while the BA and AA variables measure the added effect of getting
a diploma when graduating. Here we opted for a more simple approach."""

final_list = current + ['female','black','stotal','vlgcity','subvlg', 'nc', 'south']
### final_list = alternative + ['female','black','stotal','vlgcity','subvlg', 'nc', 'south']
### final_list = current + alternative + ['female','black','stotal','vlgcity','subvlg', 'nc', 'south']

fin_data = data[final_list]
fin_X = sm.add_constant(fin_data)
fin_model = sm.OLS(y, fin_X)
fin_results = fin_model.fit()
print(fin_results.summary())
print(fin_results.summary2().tables[1])

data_frame_to_latex_table_file(report_dir + 'results_final.tex',
                               fin_results.summary2().tables[1].round(4))

fin_white_test = sm.stats.diagnostic.het_white(fin_results.resid, fin_X)
print('F-value: {}, with p-value of {}'.format(fin_white_test[2], fin_white_test[3]))

df_f2_test = pd.Series(fin_white_test[2:], index=['F test','p-value'])

## exporting results final white test to latex table
data_frame_to_latex_table_file(report_dir + 'final_white_test.tex',
                               np.round(df_f2_test, 2))

fin_R = np.eye(len(fin_results.params))[1: len(fin_results.params)+1]

fin_hc_cov = fin_results.cov_HC0
fin_model_test = fin_results.wald_test(fin_R, fin_hc_cov)
fin_f_test = fin_results.f_test(fin_R, fin_hc_cov)
print(fin_model_test)
print(fin_f_test)

# Re-doing the confidence interval and prediction interval plot for the new model 
beta = fin_results.params
x = fin_results.model.exog

ci = np.empty((2, num_obs))
for i in range(num_obs):
    ci[0,i] = x[i].T @ beta - 1.96 * fin_results.mse_resid * np.sqrt(x[i].T @ la.inv(x.T @ x) @ x[i])
    ci[1,i] = x[i].T @ beta + 1.96 * fin_results.mse_resid * np.sqrt(x[i].T @ la.inv(x.T @ x) @ x[i])
    
pi = np.empty((2, num_obs))
for i in range(num_obs):
    pi[0,i] = x[i].T @ beta - 1.96 * fin_results.mse_resid * np.sqrt(1 + x[i].T @ la.inv(x.T @ x) @ x[i])
    pi[1,i] = x[i].T @ beta + 1.96 * fin_results.mse_resid * np.sqrt(1 + x[i].T @ la.inv(x.T @ x) @ x[i])

y_hat = x @ beta

# Plotting another graph of y against y_hat, with the confidence and prediction intervals.

fig, ax = plt.subplots(1, 1, num=fignum)
ax.plot(y_hat, y, ',')
plt.plot(y_hat, ci[0], 'r--', lw=0.25)
plt.plot(y_hat, ci[1], 'r--', lw=0.25)
plt.plot(y_hat, pi[0], 'g--', lw=0.5)
plt.plot(y_hat, pi[1], 'g--', lw=0.5)
ax.grid(True, ls=':')
ax.set_xlabel('y_hat')
ax.set_ylabel('y')
ax.set_title("Confidence and prediction intervals")
ax.legend(loc='best')
plt.savefig(figure_dir + 'empirical_figure_{}.png'.format(fignum))
plt.show()

fignum =+ 1
