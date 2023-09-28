#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 19:04:54 2022

@author: Max Temmerman and Marcio Reverbel
"""

import os
import numpy as np
import numpy.linalg as la
import pandas as pd
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import math

from helper import print_question, data_frame_to_latex_table_file

#------------------------------------------------------------------------------
def beta_hist(beta_hat, beta_avg, beta_std, num_std, i, j, fignum):
    """ 
    This is a function to print the histgrams for the OLS estimates more easily.
    It helps to iterate through each of the error scenarios and each of the coefficients.    
    """
    text = [r'$\hat{\beta}$ with (n = 10, ${\epsilon}$ ~ N(0, ${\sigma^2}$))', 
        r'$\hat{\beta}$ with (n = 100, ${\epsilon}$ ~ N(0, ${\sigma^2}$))',
        r'$\hat{\beta}$ with (n = 10, ${\epsilon}$ ~ Laplace(0, ${\sigma^2}$))',
        r'$\hat{\beta}$ with (n = 100, ${\epsilon}$ ~ Laplace(0, ${\sigma^2}$))']


    fig, ax = plt.subplots(1, 1, num=fignum)
    bins = np.linspace(beta_avg[i][j] - num_std * beta_std[i][j], beta_avg[i][j] + num_std * beta_std[i][j], 100)
    ax.hist(beta_hat[i, j], bins, density=True)
    ax.set_xlabel(r'$\beta_{}$'.format(i))
    ax.set_title(text[j])
    plt.savefig(figure_dir + 'figure_{}.png'.format(fignum))
    plt.show()

    return fignum + 1

#-----------------------------------------------------------------------------
def t_hist(t_val, t_avg, t_std, num_std, i, j, fignum):
    """ 
    This is a function to print the histgrams of the t_values more easily.
    It helps to iterate through each of the error scenarios and each of the coefficients.    
    """
    text = [r't-values with (n = 10, ${\epsilon}$ ~ N(0, ${\sigma^2}$))', 
        r't-values with (n = 100, ${\epsilon}$ ~ N(0, ${\sigma^2}$))',
        r't-values with (n = 10, ${\epsilon}$ ~ Laplace(0, ${\sigma^2}$))',
        r't-values with (n = 100, ${\epsilon}$ ~ Laplace(0, ${\sigma^2}$))']

    fig, ax = plt.subplots(1, 1, num=fignum)
    bins = np.linspace(t_avg[i][j] - num_std * t_std[i][j], t_avg[i][j] + num_std * t_std[i][j], 100)
    ax.hist(t_val[i, j], bins, density=True)
    ax.plot(bins, stats.t.pdf(bins, n_obs[j]-3), lw = 2, label = 't')
    ax.plot(bins, stats.norm.pdf(bins), label = 'N') 
    ax.grid(True, ls=':')
    ax.set_xlabel(r't-values for $\beta_{}$'.format(i))
    ax.set_title(text[j])
    ax.legend(loc='best')
    plt.savefig(figure_dir + 'figure_{}.png'.format(fignum))
    plt.show()
    
    return fignum + 1

#------------------------------------------------------------------------------
def f_hist(f_val, f_avg, f_std, num_std, i, fignum):
    """ 
    This is a function to print the histgrams of the f_values more easily.
    It helps to iterate through each of the error scenarios.    
    """

    text = [r'Model test (n = 10, ${\epsilon}$ ~ N(0, ${\sigma^2}$))', 
        r'Model test (n = 100, ${\epsilon}$ ~ N(0, ${\sigma^2}$))',
        r'Model test (n = 10, ${\epsilon}$ ~ Laplace(0, ${\sigma^2}$))',
        r'Model test (n = 100, ${\epsilon}$ ~ Laplace(0, ${\sigma^2}$))']
    lab = [r'${F_{2,7}}$', r'${F_{2,97}}$', r'${F_{2,7}}$', r'${F_{2,97}}$', r'${X{^2_2}}$']
    fig, ax = plt.subplots(1, 1, num=fignum)
    
    # IMPORTANT: These graphics don't construct well for n = 10 because of the enourmous difference 
    # in scale between the histogram data and the distributions. If you change the number of bins below, from
    # 150 to 500, for example, it's possible to see the distribution in those two cases (Laplace and Normal,
    # with n = 10), but the histograms don't look good. Basically, now the bins are too small to show the portion
    # of the pdf that is farther from the horizontal line (x = 0).
    bins = np.linspace(f_avg[i] - num_std * f_std[i], f_avg[i] + num_std * f_std[i], 150)
    ax.hist(f_val[i], bins, density=True)
    ax.plot(bins, stats.f.pdf(bins, 2, n_obs[i]-2), label=lab[i])
    ax.plot(bins, stats.chi2.pdf(bins, 2), label = lab[4])    
    ax.grid(True, ls=':')
    ax.set_xlabel(r'f-values for the model test: $\beta_0$ = $\beta_1$ = $\beta_2$ = 0')
    ax.set_title(text[i])
    ax.legend(loc='best')
    plt.savefig(figure_dir + 'figure_{}.png'.format(fignum))
    plt.show()

    return fignum + 1

#-----------------------------------------------------------------------------
def eps_hist(eps, j, fignum):
    """ 
    This is a function to print the histgrams of the error terms more easily.
    It helps to iterate through each of the error types and each of the coefficients.    
    """
    tit = [r'${\epsilon}$ ~ N(0, 0.25${^2}$), n = 10',
           r'${\epsilon}$ ~ N(0, 0.25${^2}$), n = 100',
           r'${\epsilon}$ ~ Laplace(0, 0.25${^2}$), n = 10',
           r'${\epsilon}$ ~ Laplace(0, 0.25${^2}$), n = 100']
    fig, ax = plt.subplots(1, 1, num=fignum)
    ax.hist(eps.flatten(), 100, density=True)
    ax.set_xlabel(r'${\epsilon}$')
    ax.set_title(tit[j])
    plt.savefig(figure_dir + 'figure_{}.png'.format(fignum))
    plt.show()
    
    return fignum + 1

#------------------------------------------------------------------------------
# Adjusting the working directory
cwd = os.getcwd()
print('Current Working Directory is: ', cwd, '\n')
"""
absolute_path = '../python'
os.chdir(absolute_path)
print('New working directory is: ', os.getcwd(), '\n')
"""

#------------------------------------------------------------------------------
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

#------------------------------------------------------------------------------
# Code starts here! 
#------------------------------------------------------------------------------

bd_1 = 1106
bd_2 = 706
group_seed = bd_1*bd_2

rng = np.random.default_rng(group_seed)
fignum = 1
""" QUESTION 1 - Creating the matrix of observations """

mu = {}
mu["x1"] = 5
mu["x2"] = 0

std = {}
std["x1"] = 1
std["x2"] = 2

rho = 0.5

# building the covariance matrix between x1 and x2:
cov = np.array([[pow(std["x1"], 2), 0.5*std["x1"]*std["x2"]],[0.5*std["x1"]*std["x2"], pow(std["x2"], 2)]])

x1, x2 = rng.multivariate_normal([mu["x1"], mu["x2"]], cov, 100).T
x = np.vstack([np.ones(100), x1, x2]).T 

""" QUESTION 2 - Generating error terms for each observation """ 
#------------------------------------------------------------------------------
# The real values of the coefficients. This is known only to the "creator".
beta = np.array([10, 0.5, 2.2])
sigma= 0.25

# the number of observations was copied twice into the tuple to make it easier to iterate
# when making calculations and passing information to functions.
n_obs = (10, 100, 10, 100)
n_sim = pow(2, 16)

# Now we generate the errors and place them into two dictionaries of lists, separated by distribution.
eps = {}
eps["normal"] = [np.empty((n_obs[0], n_sim)), np.empty((n_obs[1], n_sim))]
eps["laplace"] = [np.empty((n_obs[0], n_sim)), np.empty((n_obs[1], n_sim))]

eps["normal"][0] = rng.normal(0, sigma, (n_obs[0], n_sim))
eps["normal"][1] = rng.normal(0, sigma, (n_obs[1], n_sim))
eps["laplace"][0] = rng.laplace(0, sigma/np.sqrt(2), (n_obs[0], n_sim))
eps["laplace"][1] = rng.laplace(0, sigma/np.sqrt(2), (n_obs[1], n_sim))

""" QUESTION 3 - Generating the dependent variable """
#-----------------------------------------------------------------------------
# We create the X for 10 observations and the X for 100 observations in the temp variable. 
temp = [x[0:10, :] @ beta, x @ beta]

y = [np.empty((n_obs[0], n_sim)), np.empty((n_obs[1], n_sim)), np.empty((n_obs[2], n_sim)), np.empty((n_obs[3], n_sim))]

# We create the y variable using the appropriate errors and x (temp).
y[0] = np.add(eps["normal"][0].T, temp[0]).T
y[1] = np.add(eps["normal"][1].T, temp[1]).T
y[2] = np.add(eps["laplace"][0].T, temp[0]).T
y[3] = np.add(eps["laplace"][1].T, temp[1]).T

#------------------------------------------------------------------------------
""" QUESTION 4 - Plotting the error terms """
# Here we iterate through all simulation scenarios to create the relevant histograms. 
# See the eps_hist function for more details.
i = 0
for value in eps.values():
    fignum = eps_hist(value[0], i, fignum)
    fignum = eps_hist(value[1], i + 1, fignum)
    i =+ 2
""" QUESTION 5 - Calculate the OLS estimates, t-tests, and model test """

# We created a lambda function that generates the regression with inputs being simulation scenario and simulation number.

# It works, but extremely slow. To make the code faster, we've calculated the OLS estimates and t-values manually. 
# The code for using the OLS.results.params and results.t_test is there, but commented. If you wish to use it,
# un-comment all the lines starting with '###'. It will take some 7 minutes to run. 
# The np.isclose() function was used to compare the results from manual calculations and the functional outputs
# for the OLS estimates and t values. All values match.
 
r = lambda i, s: sm.OLS(y[i][:, s], x[:n_obs[i]]).fit()

###beta_OLS = np.empty((3, 4, n_sim))
###t_val_OLS = np.empty((3, 4, n_sim))
f_val = np.empty((4, n_sim))
for s in range(n_sim):
        for i in range(4):
            results = r(i, s)
###            beta_OLS[0, i, s], beta_OLS[1, i, s], beta_OLS[2, i, s] = results.params
###            t_val_OLS[0, i, s], t_val_OLS[1, i, s], t_val_OLS[2, i, s], = results.t_test('const = 10, x1 = 0.5, x2 = 2.2').tvalue
            f_val[i, s] = results.fvalue

# Calculating the estimates manually
beta_hat = {}
beta_hat["normal"] = [np.empty((3, n_sim)).T, np.empty((3, n_sim)).T]
beta_hat["laplace"] = [np.empty((3, n_sim)).T, np.empty((3, n_sim)).T]

for i in range(0, n_sim): 
    beta_hat["normal"][0][i] = la.inv(x[0:10].T @ x[0:10]) @ x[0:10].T @ y[0][:, i]
    beta_hat["normal"][1][i] = la.inv(x.T @ x) @ x.T @ y[1][:, i]
    beta_hat["laplace"][0][i] = la.inv(x[0:10].T @ x[0:10]) @ x[0:10].T @ y[2][:, i]
    beta_hat["laplace"][1][i] = la.inv(x.T @ x) @ x.T @ y[3][:, i]

sim = (beta_hat["normal"][0], beta_hat["normal"][1], beta_hat["laplace"][0], beta_hat["laplace"][1])

b0 = np.array([sim[0][:, 0], sim[1][:, 0], sim[2][:, 0], sim[3][:, 0]])
b1 = np.array([sim[0][:, 1], sim[1][:, 1], sim[2][:, 1], sim[3][:, 1]])
b2 = np.array([sim[0][:, 2], sim[1][:, 2], sim[2][:, 2], sim[3][:, 2]])

b_hat = np.array([b0, b1, b2])
###print(np.isclose(beta_OLS, b_hat))
###print('\n')
#------------------------------------------------------------------------------
# Calculating the t-tests manually

# Here we develop a lambda function that allows us to easily calculate y_hat for each  simulation scenario
y_hat = (lambda i : x[0:n_obs[i]] @ [b0[i], b1[i], b2[i]])

e = [np.empty((n_obs[0], n_sim)), np.empty((n_obs[1], n_sim)), np.empty((n_obs[0], n_sim)), np.empty((n_obs[1], n_sim))]

for i in range(4):
    e[i] = y[i] - y_hat(i)

t_val = np.empty((3, 4, n_sim))
for i in range(4):
    for j in range(3):
        """ Here we are calculating the t-value with the formula learned in class: t = (B_hat - Bnull)/SE(index j)
        with SE = 1/sqrt(n-k) * e'e * sqrt(X'X). It looks slightly more complicated because it is being 
        calculated for all betas (j) and simulation scenarios (i).""" 
        t_val[j, i] = (b_hat[j][i] - beta[j])/np.sqrt(np.sum(e[i]**2, 0)/(n_obs[i]-3)*
                                                  la.inv(x[0:n_obs[i]].T @ x[0:n_obs[i]])[j, j])
    #OBS: degrees of freedom = n - k - 1, n_obs - number of independent variables - 1. 

###print(np.isclose(t_val_OLS, t_val))

print("b_hat dimensions: {}\nt_val dimensions: {}\nf_val dimensions: {}".format(b_hat.shape, t_val.shape, f_val.shape))

#------------------------------------------------------------------------------
""" QUESTION 6, 7, 8 - Creating histograms for each of the quantities found, 
and superposing the t, f, normal and chi-square distributions over the relevant 
histograms. Consult the functions at the top of the document for more information.
"""

# calculate the average of all the OLS estimates
beta_avg = np.array([np.mean(b_hat[0], 1), np.mean(b_hat[1], 1), np.mean(b_hat[2], 1)])
# same for the standard deviation
beta_std = np.array([np.std(b_hat[0], 1), np.std(b_hat[1], 1), np.std(b_hat[2], 1)])
# used for the bins in the histogram later on
num_std = 3.5

# create figures:
# Using the beta_hist function to plot all 12 histograms for the OLS estimates
for i in range(3):
    for j in range(4):
        fignum = beta_hist(b_hat, beta_avg, beta_std, num_std, i, j, fignum)

# calculate the average of all the OLS estimates
t_avg = np.array([np.mean(t_val[0], 1), np.mean(t_val[1], 1), np.mean(t_val[2], 1)])
# same for the standard deviation
t_std = np.array([np.std(t_val[0], 1), np.std(t_val[1], 1), np.std(t_val[2], 1)])

# Using the t_hist function to plot all 12 histograms for the OLS estimates
for i in range(3):
    for j in range(4):
        fignum = t_hist(t_val, t_avg, t_std, num_std, i, j, fignum)

# Using the f_hist function to plot all 4 histograms of the F values:                
f_avg = np.mean(f_val, 1)
f_std = np.std(f_val, 1)

for i in range(4):
    fignum = f_hist(f_val, f_avg, f_std, num_std, i, fignum)
#------------------------------------------------------------------------------

