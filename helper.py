#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper functions for assignment
"""
import os
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# ALL FUNCTIONS ARE ON TOP
#
# THE SCRIPT IS BELOW THE FUNCTIONS
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# Helper function to transform results summary into a dataFrame
# -----------------------------------------------------------------------------

def results_summary_to_dataframe(results, rounding=2):
    '''take the result of an statsmodel results table
    and transforms it into a dataframe'''

    # get the values from results
    # if you want, you can of course generalize this.
    # e.g. if you don't have normal error terms
    # you could change the pvalues and confidence bounds
    # see exercise session 9?!
    pvals = results.pvalues
    tvals = results.tvalues
    coeff = results.params
    conf_lower = results.conf_int()[:, 0]
    conf_higher = results.conf_int()[:, 1]

    # create a pandas DataFrame from a dictionary
    results_df = pd.DataFrame({"pvals": np.round(pvals, rounding),
                               "tvals": np.round(tvals, rounding),
                               "coeff": np.round(coeff, rounding),
                               "conf_lower": np.round(conf_lower, rounding),
                               "conf_higher": np.round(conf_higher, rounding)
                               })
    # This is just to show you how to re-order if needed
    # Typically you should put them in the order you like straigh away
    #Reordering...
    results_df = results_df[["coeff", "tvals", "pvals", "conf_lower",
                             "conf_higher"]]

    return results_df

# -----------------------------------------------------------------------------


def data_frame_to_latex_table_file(file_name, df):
    """takes a DataFrame and creates file_name.tex with LaTeX table data. """
    # create and open file
    text_file = open(file_name, "w")
    # data frame to LaTeX
    df_latex = df.to_latex()
    # Consider extensions (see later in class)
    # write latex string to file
    text_file.write(df_latex)
    # close file
    text_file.close()

# -----------------------------------------------------------------------------


def print_question(statement, print_line_start=5, print_line_length=90):
    """
    Print question description.

    Args:
        statement (string): Question description.
        print_line_start (int): column where statement starts
        print_line_length (int): total columns.

    Returns:
        None.

    """
    print(print_line_start * '#' + ' ' + statement + ' ' +
          (print_line_length - len(statement) - print_line_start - 2) * '#')

