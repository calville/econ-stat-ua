

# =============================================================================
# Analysis of the relationship between 'Per capita gross regional product' and other various regional indicators in Ukraine.
# =============================================================================


# Load all needed modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import RFE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


# All data gathered on the official site of National Statistics Service of Ukraine
# Import dataset with different indicators by region (without Crimea and  occupied areas of Donetsk and Luhansk regions)
df = pd.read_excel('https://github.com/calville/econ-stat-ua/raw/8cca7c38bb3a089d00b6c43e50ce05caa3e17b05/ds-stat.xls', sheet_name=0, header=2, skipfooter=1)

# Check data types, indexes, NaN values
df.info()
df.index.names
df.columns
df.head(10)

# Export quick statistics
# df.describe().to_csv('describe.csv')
df['grp-capita'].describe()





# ---------------------------------------------------------------------------
### Data Wrangling

# Replace missing values with NaN
df.replace(' ', np.nan, inplace = True)

# Detect missing data
missing_data = df.isnull()
missing_data.head(5)

# Figure out columns with missing values
for column in missing_data.columns.values.tolist():
    if missing_data[column].value_counts()[0]!=25:
        print(column)
        print(missing_data[column].value_counts())
        print('')

# Calculate the average of columns with NaN values
avg_apartm_new_rural = df['apartm-new-rural'].median()
print('Median average of New construction in rural areas:', avg_apartm_new_rural)

avg_parks = df['parks'].median()
print('Median average of National nature parks area:', avg_parks)

avg_child_pre_sch = df['child-pre-sch'].median()
print('Median average of child-pre-sch:', avg_child_pre_sch)


# Replace missing values like NaNs with actual values
df['apartm-new-rural'].replace(np.nan, avg_apartm_new_rural, inplace=True)
df['parks'].replace(np.nan, avg_parks, inplace=True)
df['child-pre-sch'].replace(np.nan, avg_child_pre_sch, inplace=True)

# Check the result in column 'Non-Null Count'
df.info()





## Data Normalization
# Normalization is the process of transforming values of several variables into a similar range.

# Convert all monetary units to us dollars
varlist = ['grp', 'grp-capita', 'inc-full', 'inc-disp', 'outgo', 'inc-capita', 'invest', 'invest-capita', 'constr-prod', 'prom-real', 'petrol-sales-earn', 'diesel-sales-earn', 'prop-sales-earn', 'forest-cost', 'forest-unit', 'travel-legal-earn', 'travel-entpre-earn']

df[varlist] = round(df[varlist]/26.678, 2)
		
# We will use grp-capita as target variable, let's remove other duplicating columns
df.drop(['inc-full', 'inc-disp', 'outgo', 'inc-capita', 'inc-share-prev', 'grp-share'], axis = 1, inplace=True)


# Scale all our potential numeric independent variables with simple feature scaling method
df1 = df.copy()

for column in df1.columns.values.tolist():
    target = ['grp', 'grp-capita', 'region-eng']
    if column in target:
        pass
    else: df1[column] = round(df1[column]/df1[column].max(), 2)
        




## Binning
# Binning is a process of transforming continuous numerical variables into discrete categorical 'bins', for grouped analysis.

# For instance lets group 'Enterprises' and 'GRP per capita' variables into bins

# Exclude Kyiv region because it excessively distort final result
df2 = df.drop(df[df['region-eng'] == 'Kyiv'].index)
df2.reset_index(drop=True, inplace=True)
df2.info()


# Binning of independent variable 'Enterprises' (ent)

# Lets plot the histogram of Enterprises (ent), to see what the distribution of ent looks like.
plt.hist(df2['ent'])
# set x/y labels and plot title
plt.xlabel('Enterprises')
plt.ylabel('count')
plt.title('Enterprises bins')


# We would like 3 bins of equal size bandwidth so we use numpy's linspace  function.
bins = np.linspace(min(df2['ent']), max(df2['ent']), 4)
bins

# Set group names
group_names = ['Low', 'Medium', 'High']

# Apply the function 'cut' to determine what each value of variable belongs to
df2['ent-binned'] = pd.cut(df2['ent'], bins, labels=group_names, include_lowest=True)
df2[['ent','ent-binned']]

# Lets see the number of regions in each bin.
df2['ent-binned'].value_counts()

# Regions with medium and high number of Enterprises
df2[['region-eng', 'grp-capita', 'ent-binned']].loc[df2['ent-binned'].isin(['High','Medium'])].sort_values(by=['grp-capita'], ascending=False)
# Here we see that most businesses are concentrated in 7 regions.


# Lets plot the distribution of each bin.
plt.bar(df2['ent-binned'].value_counts().index, df2['ent-binned'].value_counts())
plt.xlabel('Enterprises')
plt.ylabel('count')
plt.title('Enterprises bins')



# Binning of target variable (GRP per capita) the same way like we do it for 'Enterprises' (also without the city of Kyiv)
bins = np.linspace(df2['grp-capita'].min(), df2['grp-capita'].max(), 4)
group_names = ['Low', 'Medium', 'High']
df2['grp-capita-binned'] = pd.cut(df2['grp-capita'], bins, labels = group_names, include_lowest=True)
df2['grp-capita-binned'].value_counts()
df2[['region-eng','grp-capita','grp-capita-binned']]
df2[['region-eng', 'grp-capita', 'grp-capita-binned']].loc[df2['grp-capita-binned'].isin(['High','Medium'])].sort_values(by=['grp-capita'], ascending=False)
# We have 3 regions in a group with conditionally high 'GRP per capita'. It's Poltavska, Kyivska, Dnipropetrovska

# Lets plot the histogram of grp per capita, to see what the distribution of 'grp-capita' looks like.
plt.hist(df2['grp-capita'])
plt.xlabel('grp-capita')
plt.ylabel('count')
plt.title('grp-capita bins')

# Lets plot the distribution of each bin.
plt.bar(df2['grp-capita-binned'].value_counts().index, df2['grp-capita-binned'].value_counts())
plt.xlabel('grp-capita')
plt.ylabel('count')
plt.title('grp-capita bins')





# ---------------------------------------------------------------------------
### Exploratory Data Analysis

## Correlation
# Now we find out to what extent all our variables are interdependent, using correlation coefficient.

# Calculate the correlation (Pearson coefficient) between variables of type 'float64' using the method 'corr'
corr = df1.corr()
# corr.to_csv('corr.csv')

# In the output table cosider correlation coefficients for target variable (column 'grp-capita'):

    # 1. Indicators with values of r coefficient within range (-0.25:0.25) have no relationship with target variable
norel = corr.loc[abs(corr['grp-capita']) < 0.25, 'grp-capita']
norel
len(norel)
norel.index
norel.values
list(zip(norel.index, round(norel, 4)))
        # This group consists of 17 variables: 'ent-agro', 'ent-govern', 'apartm-new-rural', 'area-agro', 'area-built', 'area-water', 'area-wet', 'parks', 'water-cons', 'waste-3', 'waste-4', 'area-harvest', 'forest-quant', 'forest-cost', 'fish', 'child-pre-sch', 'aids'.

# All our indicators are continuous numerical variables. Let's visualize some of these variables as potential predictors of 'grp-capita' using scatterplots with fitted lines. Additionally it's a convenient way to see scatter of values on the graph and outliers in the sample.
sns.regplot(x='ent-agro', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='waste-3', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='area-harvest', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='forest-quant', y='grp-capita', data=df)
plt.ylim(0,)
# Fitted lines in this graphs are close to horizontal and thus independent variables aren't good predictors of the 'Per capita gross regional product'


    # 2. Indicators with values of r coefficient within range (-0.45:-0.25) and (0.25:0.45) have  weak downhill (negative) or weak uphill (positive) linear relationship with target variable respectively
weak_rel = corr.loc[(abs(corr['grp-capita']) > 0.25) & (abs(corr['grp-capita']) < 0.45), 'grp-capita']
weak_rel
len(weak_rel)
weak_rel.index
weak_rel.values
list(zip(weak_rel.index, round(weak_rel, 4)))
        # This group consists of 10 variables: 'pop-total', 'pop-urban', 'pop-rural', 'area', 'area-wood', 'area-other', 'unemp', 'hiv', 'tub', 'pens-disab'.

# Let's visualize some variables by using scatterplots with fitted lines.
sns.regplot(x='pop-total', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='area', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='pens-disab', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='unemp', y='grp-capita', data=df)
plt.ylim(0,)
# Indicators in this group have weak correlation with target variable and couldn't be such a good predictors of the 'Per capita gross regional product'


    # 3. Indicators with values of r coefficient within range (-0.7:-0.45) and (0.45:0.7) have  moderate downhill (negative) or moderate uphill (positive) linear relationship with target variable respectively
mod_rel = corr.loc[(abs(corr['grp-capita']) > 0.45) & (abs(corr['grp-capita']) < 0.7), 'grp-capita']
mod_rel
len(mod_rel)
mod_rel.index
mod_rel.values
list(zip(mod_rel.index, round(mod_rel, 4)))
        # This group consists of 20 variables: 'share-urban', 'share-rural', 'ent-edu', 'entpre', 'apartm-area', 'flats-num', 'apartm-new', 'transp-freight', 'prom-real', 'diesel-sales-earn', 'prop-sales-earn', 'diesel-sales-count', 'prop-sales-count', 'wheat-yield', 'forest-unit', 'travel-entpre-earn', 'immigr', 'unemp-share-act', 'pens', 'crimes-drugs'.

# Let's visualize some variables by using scatterplots with fitted lines.
sns.regplot(x='share-urban', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='entpre', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='transp-freight', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='prom-real', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='wheat-yield', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='travel-entpre-earn', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='immigr', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='unemp-share-act', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='pens', y='grp-capita', data=df)
plt.ylim(0,)
# Indicators in this group have moderate correlation with target variable and could be predictors of the 'Per capita gross regional product'


    # 4. Indicators with values of r coefficient less than -0.7 and more than 0.7 have  strong downhill (negative) or strong uphill (positive) linear relationship with target variable respectively
str_rel = corr.loc[abs(corr['grp-capita']) > 0.7, 'grp-capita']
str_rel
len(str_rel)
str_rel.index
str_rel.values
list(zip(str_rel.index, round(str_rel, 4)))
        # This group consists of 44 variables: 'grp', 'grp-capita', 'ent', 'ent-ind', 'ent-constr', 'ent-trade', 'ent-transp', 'ent-accom', 'ent-info', 'ent-fin', 'ent-re', 'ent-sci', 'ent-serv', 'ent-health', 'ent-art', 'ent-other', 'goods-exp', 'goods-exp-share', 'goods-imp', 'goods-imp-share', 'goods-saldo', 'serv-exp', 'serv-exp-share', 'serv-imp', 'serv-imp-share', 'serv-saldo', 'invest', 'invest-share', 'invest-capita', 'apartm-new-urban', 'transp-vol', 'constr-prod', 'petrol-sales-earn', 'petrol-sales-count', 'inet-users', 'travel-legal', 'travel-entpre', 'travel-legal-earn', 'emigr', 'doct', 'paramed', 'crimes', 'crimes-heavy', 'crimes-group'.

# Let's visualize some variables by using scatterplots with fitted lines.
sns.regplot(x='ent', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='goods-exp', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='goods-imp', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='goods-saldo', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='serv-exp', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='serv-saldo', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='invest-capita', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='invest', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='transp-vol', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='constr-prod', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='petrol-sales-earn', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='petrol-sales-count', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='inet-users', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='travel-legal', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='travel-entpre', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='doct', y='grp-capita', data=df)
plt.ylim(0,)
sns.regplot(x='crimes', y='grp-capita', data=df)
plt.ylim(0,)
# Indicators in this group have strong relationship with target variable and would be good predictors of the 'Per capita gross regional product'


# Using Pearson coefficients and graphical analysis data select suitable statistical indicators for predicting model:
# 'ent', 'goods-exp', 'goods-imp', 'invest', 'invest-capita', 'transp-vol', 'constr-prod', 'petrol-sales-count', 'travel-entpre', 'doct', 'crimes', 'immigr'.



# P-value
# Calculate P-value (probability value) to find out if the correlation between predictor and response variables is statistically significant.
# A significance level of 0.05, which means that we are 95% confident that the correlation between the variables is statistically significant.

pearson_coef, p_value = stats.pearsonr(df1['ent'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Number of enterprises in Ukraine' and 'Gross regional product per capita' is statistically significant, and the linear relationship is very strong (~0.91)

pearson_coef, p_value = stats.pearsonr(df1['goods-exp'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Goods exports' and 'Gross regional product per capita' is statistically significant, and the linear relationship is quite strong (~0.867)

pearson_coef, p_value = stats.pearsonr(df1['goods-imp'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Goods imports' and 'Gross regional product per capita' is statistically significant, and the linear relationship is very strong (~0.931)

pearson_coef, p_value = stats.pearsonr(df1['invest'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Capital investments' and 'Gross regional product per capita' is statistically significant, and the linear relationship is very strong (~0.934)

pearson_coef, p_value = stats.pearsonr(df1['invest-capita'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Capital investments per person' and 'Gross regional product per capita' is statistically significant, and the linear relationship is very strong (~0.977)

pearson_coef, p_value = stats.pearsonr(df1['transp-vol'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Volume of freight road transport by region' and 'Gross regional product per capita' is statistically significant, and the linear relationship is quite strong (~0.755)

pearson_coef, p_value = stats.pearsonr(df1['constr-prod'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Volume of construction production' and 'Gross regional product per capita' is statistically significant, and the linear relationship is quite strong (~0.805)

pearson_coef, p_value = stats.pearsonr(df1['petrol-sales-count'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Volume wholesale and retail trade of Petrol total from the petrol stations' and 'Gross regional product per capita' is statistically significant, and the linear relationship is quite strong (~0.798)

pearson_coef, p_value = stats.pearsonr(df1['travel-entpre'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Number of travel agents - individuals - entrepreneurs ' and 'Gross regional product per capita' is statistically significant, and the linear relationship is moderately strong (~0.731)

pearson_coef, p_value = stats.pearsonr(df1['doct'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Physicians of all specializations' and 'Gross regional product per capita' is statistically significant, and the linear relationship is quite strong (~0.804)

pearson_coef, p_value = stats.pearsonr(df1['crimes'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Number of Detected crimes' and 'Gross regional product per capita' is statistically significant, and the linear relationship is quite strong (~0.793)

pearson_coef, p_value = stats.pearsonr(df1['immigr'], df1['grp-capita'])
print('The Pearson Correlation Coefficient is', pearson_coef, ' with a P-value of P =', '{:.25f}'.format(p_value))
# Since the p-value is < 0.001, the correlation between 'Number of Total immigrants' and 'Gross regional product per capita' is statistically significant, and the linear relationship is moderate (~0.656)



# We now have a better idea of what our data looks like and which variables are important to take into account when predicting the 'Gross regional product per capita'. We have narrowed it down to the following variables: 'ent', 'goods-exp', 'goods-imp', 'invest-capita', 'transp-vol', 'constr-prod', 'petrol-sales-count', 'travel-entpre', 'doct', 'crimes', 'immigr'







# ---------------------------------------------------------------------------
### Model Development, Evaluation and Refinement

# Now we can develop several models that will predict the 'Gross regional product per capita' using other features as predictors.

### Linear Regression

# LinearRegression often fitted using the least squares approach, which fits a linear model with coefficients to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted by the linear approximation.

# Let's use factor with the highest Pearson coefficient ('Capital investments per person') for our Linear Regression model to look at how this feature can help us predict 'Per capita gross regional product'
X = df1[['invest-capita']]
Y = df1['grp-capita']

# Create the linear regression object
lm = LinearRegression()
lm

# Fit the linear model using feature 'invest-capita'.
lm.fit(X,Y)

# Output a prediction 
Yhat=lm.predict(X)
Yhat[0:5]

# The values of the Intercept and the Slope
lm.intercept_
lm.coef_

# Plugging in the actual values we get final equation of our  Linear Regression model:
# 'grp-capita' = 1249.44 + 10811.68 * 'invest-capita'




## Model Evaluation
# Now we have to evaluate our model. One way to do this is by using visualization.

# Initially we have to check if the error terms are normally distributed (which is one of the major assumptions of linear regression)

# Plot the histogram of the error terms
fig = plt.figure()
sns.distplot((Y - Yhat), bins = 5)
fig.suptitle('Error Terms', fontsize = 20)
plt.xlabel('Errors', fontsize = 18)
# As we can see, the error terms resemble closely to a normal distribution. 


# As we have already found out, Regression Plot show a combination of a scattered data points (a scatter plot), as well as the fitted linear regression line going through the data.
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.regplot(x='invest-capita', y='grp-capita', data=df)
plt.ylim(0,)

# A good way to visualize the variance of the data is to use a Residual Plot.
# Randomly spread out residuals means that the variance is constant, and thus the linear model is a good fit for this data.
width = 12
height = 10
plt.figure(figsize=(width, height))
sns.residplot(df['invest-capita'], df['grp-capita'])
plt.show()
# We can see from this residual plot that the residuals are not randomly spread around the x-axis, which leads us to believe that maybe a non-linear model is more appropriate for this data.



# For quantitative evaluation of our models we can use two measures:
# 1) R^2 / R-squared (coefficient of determination), is a measure to indicate how close the data is to the fitted regression line.
# 2) Mean Squared Error (MSE) measures the average of the squares of errors, that is, the difference between actual value (y) and the estimated value (ŷ).

# Find the R^2
print('The R-square is: ', lm.score(X, Y))
# We can say that ~ 95.43% of the variation of the 'grp-capita' is explained by this simple linear model 'invest-capita'.

# Calculate the MSE
# We compare the predicted results with the actual results
mse = mean_squared_error(df1['grp-capita'], Yhat)
print('The mean square error of price and predicted value is: ', mse)





## Training and Testing
# Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model would fail to predict anything useful on yet-unseen data.
# An important step in testing your model is to split your data into training and testing data.
# We will place the target data 'grp-capita' in a separate dataframe:
y_data = df1['grp-capita']

# Drop 'grp-capita' data in x_data
x_data=df1.drop('grp-capita',axis=1)

# Now we randomly split our data into training and testing data using the function train_test_split from scikit-learn package.
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

# Check result of splitting
print('number of test samples :', x_test.shape[0])
print('number of training samples:',x_train.shape[0])

# We create a Linear Regression object
lre=LinearRegression()

# We fit the model using the feature 'invest-capita'
lre.fit(x_train[['invest-capita']], y_train)

# Let's Calculate the R^2 on the train data
lre.score(x_train[['invest-capita']], y_train)

# Let's Calculate the R^2 on the test data
lre.score(x_test[['invest-capita']], y_test)
# We can see the R^2 is much smaller using the test data

# For a more accurate assessment of the model let's calculate R^2 for different sizes of the test sample and random_state parameter.
Rsqu_tr = {}
Rsqu_t = {}
test_vol = [i/10 for i in range(1,6)]
random_st = [i for i in range(1,11)]
lre = LinearRegression()
for portion in test_vol:
    for rs in random_st:
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=portion, random_state=rs)
        lre.fit(x_train[['invest-capita']], y_train)
        Rsqu_tr[(portion,rs)] = lre.score(x_train[['invest-capita']], y_train)
        Rsqu_t[(portion,rs)] = lre.score(x_test[['invest-capita']], y_test)
        
# Calculate average of R^2 coefficients for our training and test samples.
len(Rsqu_tr), len(Rsqu_tr)
np.array(list(Rsqu_tr.values())).mean()
np.array(list(Rsqu_t.values())).mean()

# Here we see that mean value of R^2 coefficient for test sample is 0.705, which is much less optimistic compared to the estimation on the train data (0.95).





## Cross-validation Score
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample. This method primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data.
# It is a popular method because it generally results in a less biased  estimate of the model skill than other methods, such as a simple train/test split.

# To use Cross-validation we can call cross_val_score function from scikit-learn package.
# We input the object, the feature in this case 'invest-capita', the target data (y_data). The parameter 'cv' determines the number of folds; in this case 4.
lre.fit(x_data[['invest-capita']], y_data)
Rcross = cross_val_score(lre, x_data[['invest-capita']], y_data, cv=4)

# The default scoring is R^2; each element in the array has the average R^2 value in the fold:
Rcross

# We can calculate the average and standard deviation of our estimate:
print('The mean of the folds are', Rcross.mean(), 'and the standard deviation is' , Rcross.std())

# We can use negative squared error as a score by setting the parameter 'scoring' metric to 'neg_mean_squared_error'.
mse = -1 * cross_val_score(lre,x_data[['invest-capita']], y_data,cv=4,scoring='neg_mean_squared_error')
mse
print('The mean of the folds are', mse.mean(), 'and the standard deviation is' , mse.std())


# To predict the output we can use the function 'cross_val_predict'. The function splits up the data into the specified number of folds, using one fold for testing and the other folds are used for training.

# We input the object, the feature in this case 'invest-capita' , the target data y_data. The parameter 'cv' determines the number of folds; in this case 4. We can produce an output:
yhat = cross_val_predict(lre,x_data[['invest-capita']], y_data,cv=4)
yhat[0:5]


# For a more accurate assessment of the model let's calculate R^2 for different number of folds. Since our dataset has a very small number of observations, we will limit the number of folds to 5, because of high variance.
Rsqu_cv = {}
fold_num = [i for i in range(2,6)]
lre = LinearRegression()
for cv in fold_num:
    Rcross = cross_val_score(lre, x_data[['invest-capita']], y_data, cv=cv)
    Rsqu_cv[(cv)] = np.array(list(Rcross)).mean()

# Calculate average of R^2 coefficients for different number of folds.
Rsqu_cv
np.array(list(Rsqu_cv.values())).mean()

# Cross-validation gives most accurate estimate of the model. Mean value of R^2 coefficient is equal to 0.821.









# ---------------------------------------------------------------------------
### Multiple Linear Regression

# Multiple Linear Regression (MLR) is a statistical technique that uses several explanatory variables to predict the outcome of a response variable. 

# Adding more variables isn’t always helpful because the model may ‘over-fit,’ and it’ll be too complicated. This task of identifying the best subset of predictors to include in the model, among all possible subsets of predictors, is referred to as Variable selection (Feature selection).

# Here we use 'Backward selection' approach. It means that we fit a full model and slowly remove terms one at a time, starting with the term with the highest p-value. We continue this process until the 'Stopping rule' is met. 
# Typical stopping rules for explanatory modeling are:
    # P-value thresholds of 0.05 and 0.10.
    # The highest RSquare Adjusted or the lowest Root Mean Square Error. 
    # Variance Inflation Factor for explanatory variables less than 5 or sometimes less than 10.

# We can apply Backward selection approach using 'statsmodels' package

# Let's develop a model using all predictors that were selected on the previous steps.
Z = df1[['ent', 'goods-exp', 'goods-imp', 'invest-capita', 'transp-vol', 'constr-prod', 'petrol-sales-count', 'travel-entpre', 'doct', 'crimes', 'immigr']]
Y = df1['grp-capita']

# First, we’ll add all predictors to the model. Also add a constant separately (an intercept is not included by default).
Z_mlr = sm.add_constant(Z)

# Now we can fit the model using OLS (Ordinary Least Square) method present in the 'statsmodel'
mlr1 = sm.OLS(Y, Z_mlr).fit()

# The summary of all the different parameters of the regression model
mlr1.summary()
# Adjusted R-squared for MLR with all variables: 0.960

# Exporting summary to file
# write_path = 'mlr1.csv'
# with open(write_path, 'w') as f:
#     f.write(mlr1.summary().as_csv())

# If we look at the p-values of some of the variables, the values seem to be pretty high, which means they aren’t significant. That means we can drop those variables from the model.

# Before dropping the variables, we have to see the multicollinearity between the variables. We do that by calculating the VIF value (Variance Inflation Factor). It's a quantitative value that says how much the feature variables are correlated with each other.

# Creating a dataframe that will contain the names of all the feature variables and their VIFs
vif = pd.DataFrame()
vif['Features'] = Z.columns
vif['VIF'] = [variance_inflation_factor(Z.values, i) for i in range(Z.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# While dropping the variables, the first preference will go to the p-value. Also, we have to drop one variable at a time.


# 2) 
# Dropping the variable and updating the model.
# As we can see from the summary and the VIF, some variables are still insignificant. One of these variables is 'crimes', as it has a very high p-value of 0.963. Let’s go ahead and drop this variable.

# Dropping highly correlated variables and insignificant variables
Z_tr = Z.drop('crimes', axis=1)

# Build a fitted model after dropping the variable
Z_mlr = sm.add_constant(Z_tr)
mlr2 = sm.OLS(Y, Z_mlr).fit()
mlr2.summary()
# Adj. R-squared: 0.963

# Now, let’s calculate the VIF values for the new model.
vif = pd.DataFrame()
vif['Features'] = Z_tr.columns
vif['VIF'] = [variance_inflation_factor(Z_tr.values, i) for i in range(Z_tr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 3)
# Now, the variable 'immigr' has a high VIF (45.31) and a p-value (0.744). Hence, it isn’t of much use and should be dropped from the model. We’ll repeat the same process as before.
Z_tr = Z_tr.drop('immigr', 1)

# Build a new fitted model
Z_mlr = sm.add_constant(Z_tr)
mlr3 = sm.OLS(Y, Z_mlr).fit()
mlr3.summary()
# Adj. R-squared: 0.965

# Calculating the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = Z_tr.columns
vif['VIF'] = [variance_inflation_factor(Z_tr.values, i) for i in range(Z_tr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# We’ll repeat this process till every column’s p-value is <0.1, VIF is <10 and highest value of RSquare Adjusted


# 4) 
# 'goods-exp': 0.724(p-value), VIF (22.60)
Z_tr = Z_tr.drop('goods-exp', 1)

# Build a new fitted model
Z_mlr = sm.add_constant(Z_tr)
mlr4 = sm.OLS(Y, Z_mlr).fit()
mlr4.summary()
# Adj. R-squared: 0.967

# Calculating the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = Z_tr.columns
vif['VIF'] = [variance_inflation_factor(Z_tr.values, i) for i in range(Z_tr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 5) 
# 'ent': 0.368(p-value), VIF (157.89)
Z_tr = Z_tr.drop('ent', 1)

# Build a new fitted model
Z_mlr = sm.add_constant(Z_tr)
mlr5 = sm.OLS(Y, Z_mlr).fit()
mlr5.summary()
# Adj. R-squared: 0.967

# Calculating the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = Z_tr.columns
vif['VIF'] = [variance_inflation_factor(Z_tr.values, i) for i in range(Z_tr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# 6) 
# 'goods-imp': 0.36(p-value), VIF (8.84)
Z_tr = Z_tr.drop('goods-imp', 1)

# Build a new fitted model
Z_mlr = sm.add_constant(Z_tr)
mlr6 = sm.OLS(Y, Z_mlr).fit()
mlr6.summary()
# Adj. R-squared: 0.967

# Calculating the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = Z_tr.columns
vif['VIF'] = [variance_inflation_factor(Z_tr.values, i) for i in range(Z_tr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# 7) 
# 'travel-entpre': 0.219(p-value), VIF (19.34)
Z_tr = Z_tr.drop('travel-entpre', 1)

# Build a new fitted model
Z_mlr = sm.add_constant(Z_tr)
mlr7 = sm.OLS(Y, Z_mlr).fit()
mlr7.summary()
# Adj. R-squared: 0.966

# Calculating the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = Z_tr.columns
vif['VIF'] = [variance_inflation_factor(Z_tr.values, i) for i in range(Z_tr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# 8) 
# 'constr-prod': 0.45(p-value), VIF (7.88)
Z_tr = Z_tr.drop('constr-prod', 1)

# Build a new fitted model
Z_mlr = sm.add_constant(Z_tr)
mlr8 = sm.OLS(Y, Z_mlr).fit()
mlr8.summary()
# Adj. R-squared: 0.967

# Calculating the VIFs again for the new model
vif = pd.DataFrame()
vif['Features'] = Z_tr.columns
vif['VIF'] = [variance_inflation_factor(Z_tr.values, i) for i in range(Z_tr.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# At this stage we receive the apropriate p-values (< 0.1), highest Adjusted RSquare (0.967) but VIF values for two variables are still pretty high. Such a VIF level indicates high correlation and is cause for concern. It means that the coefficients may not be statistically significant with a Type II error.

# So this is how our final model looks like according to manual Backward selection approach.
# Feature variables for this model are: 'invest-capita', 'transp-vol', 'petrol-sales-count', 'doct'

# It’s time for us to go ahead and make predictions using the final model. 
Yhat8 = mlr8.predict(Z_mlr)





## Recursive Feature Elimination (RFE)

# There is another process to build the MLR model called Recursive Feature Elimination (RFE). It's an automatic process where we don’t need to select variables manually.
# We will use the LinearRegression function from sklearn for RFE

# Fit the linear model using the eleven above-mentioned variables.
Z = df1[['ent', 'goods-exp', 'goods-imp', 'invest-capita', 'transp-vol', 'constr-prod', 'petrol-sales-count', 'travel-entpre', 'doct', 'crimes', 'immigr']]
Y = df1['grp-capita']
mlr_rfe = LinearRegression()
mlr_rfe.fit(Z,Y)

# In the code, we have to provide the number of variables the RFE has to consider to build the model.
rfe = RFE(mlr_rfe, 4)
rfe = rfe.fit(Z, Y)

# List of features issued by RFE process
rfe_list = list(zip(Z.columns,rfe.support_,rfe.ranking_))
rfe_list

# As we can see, the variables showing True is essential for the model, and the False variable is not needed. If we want to add the False variable to the model, there is also a rank associated with them to add the variables in that order.


# Building Model
# Now, we build the model using statsmodel for detailed statistics.

# Creating dataframe with RFE selected variables
col = [i[0] for i in rfe_list if i[1] == True]
col
Z_rfe = Z[col]

# Adding a constant variable 
Z_rfe_c = sm.add_constant(Z_rfe)

# Running the linear model
mlr_rfe = sm.OLS(Y,Z_rfe_c).fit()
mlr_rfe.summary()

# Calculating the VIFs for this model
vif = pd.DataFrame()
vif['Features'] = Z_rfe.columns
vif['VIF'] = [variance_inflation_factor(Z_rfe.values, i) for i in range(Z_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif

# According to automatic process of RFE, optimal model with 4 features contains: 'ent', 'goods-imp', 'invest-capita', 'transp-vol'. As we see 2 variables differ from MLR model in manual process

# Let's compare results for models from 2 different methods.
# All p-values of this MLR model received with RFE are in the desired range (< 0.1), and they are even lower than in model from manual Backward selection.
# Highest Adjusted RSquare (0.964) for RFE model and its slitly lower than in previous one
# But VIF values for RFE model are significantly larger than in manual method. This indicates a high multicollinearity and using of this model can lead to Type II error.

# Hence we can conclude that the model obtained by manual Backward selection approach is better than RFE model.




## Model Evaluation

## Visualization

# First of all we have to check if the error terms are normally distributed (which is one of the major assumptions of linear regression)

# Make predictions using the final model from manual Backward selection approach
Yhat8 = mlr8.predict(Z_mlr)
# Actual values of the target variable
Y = df1['grp-capita']

# Plot the histogram of the Error terms
fig = plt.figure()
sns.distplot((Y - Yhat8), bins = 5)
fig.suptitle('Error terms', fontsize = 20)
plt.xlabel('Errors', fontsize = 18)
# As we can see, the error terms resemble closely to a normal distribution.


# To visualize a model for Multiple Linear Regression we can use the distribution plot. Let's look at the distribution of the fitted values that result from the model and compare it to the distribution of the actual values.

plt.figure(figsize=(width, height))
ax1 = sns.distplot(df['grp-capita'], hist=False, color='r', label='Actual Value')
sns.distplot(Yhat8, hist=False, color='b', label='Fitted Values' , ax=ax1)
plt.title('Actual vs Fitted Values for grp-capita')
plt.xlabel('grp-capita (in dollars)')
plt.ylabel('Proportion of grp-capita')
plt.show()
plt.close()

# We can see that the fitted values are reasonably close to the actual values, since the two distributions significantly overlap.



## Training and Testing

# As well as for SLR model let's calculate R^2 for different sizes of the test sample and random_state parameter.

# Place all predictors that were selected with Backward selection method in a  dataframe.
Z2 = df1[['invest-capita', 'transp-vol', 'petrol-sales-count', 'doct']]
# Place the target data in a separate dataframe
y_data2 = df1['grp-capita']

# Add all R^2 values of train and test sets to dictionaries
Rsqu_tr2 = {}
Rsqu_t2 = {}
test_vol = [i/10 for i in range(1,6)]
random_st = [i for i in range(1,11)]
lre2 = LinearRegression()
for portion in test_vol:
    for rs in random_st:
        x_train2, x_test2, y_train2, y_test2 = train_test_split(Z2, y_data2, test_size=portion, random_state=rs)
        lre2.fit(x_train2, y_train2)
        Rsqu_tr2[(portion,rs)] = lre2.score(x_train2, y_train2)
        Rsqu_t2[(portion,rs)] = lre2.score(x_test2, y_test2)
        
# Calculate average of R^2 coefficients for our training and test samples.
len(Rsqu_tr2), len(Rsqu_t2)
np.array(list(Rsqu_tr2.values())).mean()
np.array(list(Rsqu_t2.values())).mean()

# Here we see that mean value of R^2 coefficient for test sample is 0.718, which is much less optimistic compared to the estimation on the train data (0.975).



## Cross-validation Score
# As well as for SLR model let's calculate R^2 for different number of folds. 

# Add all R^2 values to dictionary
Rsqu_cv2 = {}
fold_num = [i for i in range(2,6)]
lre2 = LinearRegression()
for cv in fold_num:
    Rcross2 = cross_val_score(lre2, Z2, y_data2, cv=cv)
    Rsqu_cv2[(cv)] = np.array(list(Rcross2)).mean()

# Calculate average of R^2 coefficients for different number of folds.
Rsqu_cv2
np.array(list(Rsqu_cv2.values())).mean()

# Mean value of R^2 coefficient is equal to 0.828.



# We built a basic multiple linear regression model in machine learning manually and using an automatic RFE approach. Comparing evaluation results of the MLR model to our SLR model we can colclude, that MLR works slightly better in prediction of target variable.
# Mean value of R^2 coefficient for test sample for MLR is 0.718 (against 0.705 in SLR)
# For the cross-validation check, mean value of R^2 coefficient for MLR is 0.828 (against 0.821 in SLR)








# ---------------------------------------------------------------------------
### Polynomial Regression and Pipelines

# Polynomial regression (PRM) is a particular case of the general linear regression model or multiple linear regression models.
# We get non-linear relationships by squaring or setting higher-order terms of the predictor variables.

# We saw earlier the results obtained by SLR and MLR models. Let's see if we can try fitting a polynomial model to the data instead.

# Let's get the variables and fit the polynomial using scikit-learn package
x = df[['invest-capita']]
y = df['grp-capita']

# We create a PolynomialFeatures object of degree 3:
pr=PolynomialFeatures(degree=3)
pr

# Transform our input
poly = pr.fit_transform(x)

# Dimensions of the original data
x.shape
# Dimensions of the transformed data
poly.shape

# Create the linear regression object
prm = LinearRegression()

# Fit our preprocessed data to the polynomial regression model
prm.fit(poly, y)

# The values of coefficients for our model
prm.intercept_
prm.coef_

# Plugging in the actual values we get equation of our  Polynomial Regression model:
# y = 0.000004637 x^3 - 0.01129 x^2 + 12.17 x + 507.7

# Output a prediction 
Yhat_prm = prm.predict(poly)
Yhat_prm[0:5]





## Pipeline
# Data Pipelines simplify the steps of processing the data. We use the module Pipeline to create a pipeline. We also use StandardScaler as a step in our pipeline.

# We create the pipeline, by creating a list of tuples including the name of the model or estimator and its corresponding constructor.
Input=[('scale',StandardScaler()), ('polynomial', PolynomialFeatures(degree=3, include_bias=False)), ('model',LinearRegression())]

# we input the list as an argument to the pipeline constructor
pipe=Pipeline(Input)
pipe

# We can normalize the data, perform a transform and fit the model simultaneously. 
x = df[['invest-capita']]
y = df['grp-capita']
pipe.fit(x,y)

# Similarly, we can normalize the data, perform a transform and produce a prediction simultaneously
ypipe=pipe.predict(x)
ypipe[0:5]




## Model Evaluation

## Visualization

# For visualization of our PRM we can use scatter plot with fitted regression line

# Let's use the following function to plot the data:
def PlotPolly(model, poly, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(0, max(independent_variable.iloc[:, 0]), 100)
    y_new = model.predict(poly.fit_transform(pd.Series(x_new).to_frame()))
    plt.scatter(independent_variable, dependent_variabble, color = 'blue', s=5)
    plt.plot(x_new, y_new, color = 'red', linewidth=1)
    plt.title('Polynomial Fit with Matplotlib for GRP per capita')
    plt.xlabel(Name)
    plt.ylabel('Gross regional product per capita')
    plt.show()
    plt.close()

# Plotting the line based on fitted values that result from the model and scatter of the actual values
PlotPolly(prm, pr, x, y, 'invest-capita')

# Plot show that generated polynomial function 'hits' more of the data points than SLR. But let's see if our model is not too biased.





## Training and Testing

# As well as for SLR and MLR models let's calculate R^2 for different sizes of the test sample and random_state parameter. To find out optimal order for  our polinomial model we also calculate coefficient of determination for different 'degree' parameter.

# We test PRM on dataset without outlier due to high risk of overfitting. Overfitting occurs when the model fits the noise, not the underlying process.
y_data3 = df2['grp-capita']
x_data3 = df2[['invest-capita']]
lre3 = LinearRegression()

# Add all R^2 values of train and test sets to dictionaries
Rsqu_tr3 = {}
Rsqu_t3 = {}
degree = [i for i in range(2,7)]
test_vol = [i/10 for i in range(1,6)]
random_st = [i for i in range(1,11)]
for dg in degree:
    for portion in test_vol:
        for rs in random_st:
            pr3=PolynomialFeatures(degree=dg)
            poly3 = pr3.fit_transform(x_data3)
            x_train3, x_test3, y_train3, y_test3 = train_test_split(poly3, y_data3, test_size=portion, random_state=rs)
            lre3.fit(x_train3, y_train3)
            Rsqu_tr3[(dg,portion,rs)] = lre3.score(x_train3, y_train3)
            Rsqu_t3[(dg,portion,rs)] = lre3.score(x_test3, y_test3)

# Check all available number of variants
len(Rsqu_tr3), len(Rsqu_t3)

# Calculate average of R^2 coefficients for different orders of PRM for training and test samples.
Rsqu_tr_avg = {}
Rsqu_t_avg = {}
for dg in degree:
    tr = np.array(list({k:v for (k,v) in Rsqu_tr3.items() if k[0]==dg}.values())).mean()
    t = np.array(list({k:v for (k,v) in Rsqu_t3.items() if k[0]==dg}.values())).mean()
    Rsqu_tr_avg[dg] = tr
    Rsqu_t_avg[dg] = t

# View the results
Rsqu_tr_avg
Rsqu_t_avg

# Comparing outcome for both training and test samples, we can see that optimal order for out PRM would be 2, because it shows the highest rate of mean value of R^2 for test sample (0.577). Models with higher orders issue results close to 0 or even negative, which is a sign of overfitting. 





## Cross-validation Score

# As well as for SLR and MLR models let's calculate R^2 for different number of folds. And here we also calculate for different 'degree' parameter, to choose optimal order of the PRM.  

# Add all R^2 values to dictionary
Rsqu_cv3 = {}
degree = [i for i in range(2,7)]
fold_num = [i for i in range(2,6)]
for dg in degree:
    for cv in fold_num:
        pr3=PolynomialFeatures(degree=dg)
        pol3 = pr3.fit_transform(x_data3)
        Rcross3 = cross_val_score(lre3, pol3, y_data3, cv=cv)
        Rsqu_cv3[(dg, cv)] = np.array(list(Rcross3)).mean()
    
# Calculate average of R^2 coefficients for different order of the PRM.
Rsqu_cv_avg = {}
for dg in degree:
    cv = np.array(list({k:v for (k,v) in Rsqu_cv3.items() if k[0]==dg}.values())).mean()
    Rsqu_cv_avg[dg] = cv

# View the results
Rsqu_cv_avg

# Here we also see that the highest rate of R^2 coefficient (0.636) shows PRM of order 2.


# Let's compare distribution plots of fitted and actual values to estimate model efficacy.

# First make prediction using cross-validation 
x = df[['invest-capita']]
y = df['grp-capita']
pr=PolynomialFeatures(degree=2)
poly = pr.fit_transform(x)
prm = LinearRegression()
prm.fit(poly, y)
yhat3 = cross_val_predict(prm, poly, y, cv=4)

# Distribution plot for actual values and values predicted with model
width = 12
height = 10
plt.figure(figsize=(width, height))
ax1 = sns.distplot(y, hist=False, color='r', label='Actual Value')
sns.distplot(yhat3, hist=False, color='b', label='Fitted Values' )
plt.title('Actual vs Fitted Values for GRP per capita')
plt.xlabel('GRP per capita')
plt.ylabel('Proportion of GRP per capita')
plt.show()
plt.close()

# Two distributions slightly overlap, but fitted values are not so close to the actual values.


# Comparing evaluation results of the PRM to SLR and MLR models we can colclude, that PRM works much worse than previous models in prediction of target variable.
# Mean value of R^2 coefficient for test sample for PRM is 0.577 (MLR: 0.718, SLR: 0.705)
# For the cross-validation check, mean value of R^2 coefficient for PRM is 0.636 (MLR: 0.828, SLR: 0.821)





