import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
import statsmodels.formula.api as smf

st.markdown("""

# Linear Regression

TLDR; linear regression is a statistical method where a line is fitted on a dataset.

for example given dataset

""")


n = 10
mu = [0, 0]
cov = [[0.01, 0],
       [0, 0.01]]

df = pd.DataFrame(stats.multivariate_normal(mu).rvs(n),
                  columns=["x","y"])

fig, ax  = plt.subplots()

sns.scatterplot(x="x", y="y", data=df, ax=ax)

st.pyplot(fig)

fig, ax  = plt.subplots()

st.markdown("""
Fitting a line on the dataset might look something like this.
""")

sns.regplot(x="x",
            y="y",
            data=df,
            ax=ax)

st.pyplot(fig)

st.markdown(""""
What is happening behind the scenes is that there's a linear model
constructed out of the data.

But what that even mean?

You might remember from the formula for linear equation from a math
class, where there some intercept value ~ what value y will get if x is 0
and a slope that defines the angle of the line.

$$ y = mx + b $$

So in practice the linear model is just some numbers $m$ and $b$ that
we derived from the given data.

This is usually used to predict future y values from ne observations of x.

""")

results = smf.ols('x ~ y', data=df).fit()

st.text(results.summary())

st.text(dir(results))
