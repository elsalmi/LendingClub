---
title: Model Training and Tuning
notebook: Models.ipynb
nav_include: 3
---

## Contents
{:.no_toc}
*  
{: toc}



When building and testing models for real-world use, we should choose model performance metrics based on the goal and usage of the model. In our case, we will be focusing on a high precision bar where the positive outcome is a fully paid off loan and then maximizing the total number of true positives. 
<br>
We have chosen to limit ourselves to decision tree-based algorithms because they are flexible and apply to broad types of data. We have some important ordinal variables in our dataset, including loan subgrade.


**Imports and Loading Data**



```python
import requests
from IPython.core.display import HTML
styles = requests.get("https://raw.githubusercontent.com/Harvard-IACS/2018-CS109A/master/content/styles/cs109.css").text
HTML(styles)

from IPython.display import clear_output
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, classification_report


%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.5)
sns.set_style('ticks')

from os import listdir
from os.path import isfile, join
import warnings
warnings.filterwarnings('ignore')
```




```python
import os

#Google Collab Only
from google.colab import drive
drive.mount('/content/gdrive', force_remount=False)

```




```python
clean_df = pd.read_pickle('/content/gdrive/My Drive/Lending Club Project/data/Pickle/clean_df_Mon.pkl').sample(frac=.10, random_state=0)
```




```python
#clean_df = pd.read_pickle('./data/Pickle/clean_df.pkl').sample(frac=.10, random_state=0)

print(clean_df.shape)
outcome='fully_paid'

data_train, data_val = train_test_split(clean_df, test_size=.1, stratify=clean_df[outcome], random_state=99);

X_train = data_train.drop(columns=['issue_d', 'zip_code', 'addr_state', outcome])
y_train = data_train[outcome]

X_val = data_val.drop(columns=['issue_d', 'zip_code', 'addr_state', outcome])
y_val = data_val[outcome]
```


    (108744, 87)


## <font color='maroon'>Baseline Models</font>

Since we have performed feature selection using Random Forest, we will use a **single Decision Tree as our baseline with the top 15 features chosen by feature importances.**

First, for comparison purposes, we compute an accuracy score from a trival model: a model in which we simply predict all loans to have the outcome of the most-common class (i.e., predicting all loans to be fully paid).



```python
importances = ['int_rate', 'sub_grade', 'dti', 'installment', 'avg_cur_bal', 'credit_line_age', 'bc_open_to_buy', 'mo_sin_old_rev_tl_op', 'annual_inc', 'mo_sin_old_il_acct', 'tot_hi_cred_lim', 'revol_util', 'bc_util', 'revol_bal', 'tot_cur_bal', 'total_rev_hi_lim', 'total_bal_ex_mort', 'total_bc_limit', 'loan_amnt', 'total_il_high_credit_limit', 'grade', 'mths_since_recent_bc', 'total_acc', 'mo_sin_rcnt_rev_tl_op', 'num_rev_accts', 'num_il_tl', 'mths_since_recent_inq', 'mo_sin_rcnt_tl', 'num_bc_tl', 'acc_open_past_24mths', 'num_sats', 'open_acc', 'pct_tl_nvr_dlq', 'num_op_rev_tl', 'mths_since_last_delinq', 'percent_bc_gt_75', 'num_rev_tl_bal_gt_0', 'num_actv_rev_tl', 'num_bc_sats', 'num_tl_op_past_12m', 'term_ 60 months', 'num_actv_bc_tl', 'mths_since_recent_revol_delinq', 'mort_acc', 'mths_since_last_major_derog', 'tot_coll_amt', 'mths_since_recent_bc_dlq', 'mths_since_last_record', 'inq_last_6mths', 'num_accts_ever_120_pd', 'delinq_2yrs', 'pub_rec', 'verification_status_Verified', 'verification_status_Source Verified', 'emp_length_10+ years', 'purpose_debt_consolidation', 'emp_length_5-9 years', 'emp_length_2-4 years', 'home_ownership_RENT', 'home_ownership_MORTGAGE', 'purpose_credit_card', 'pub_rec_bankruptcies', 'home_ownership_OWN', 'num_tl_90g_dpd_24m', 'tax_liens', 'purpose_home_improvement', 'purpose_other', 'collections_12_mths_ex_med', 'purpose_major_purchase', 'purpose_small_business', 'purpose_medical', 'application_type_Joint App', 'purpose_moving', 'chargeoff_within_12_mths', 'delinq_amnt', 'purpose_vacation', 'purpose_house', 'acc_now_delinq', 'purpose_wedding', 'purpose_renewable_energy', 'home_ownership_OTHER', 'home_ownership_NONE', 'purpose_educational']
```




```python
top_15_features = importances[:15]
```


**For the purposes of our baseline model, we simply select the top 15 features from the random forest feature importances. We train and compare decison trees using this subset of features. **

**In subsequent models, we will engineer new features and carefully tune the subset of features to include.**




```python
most_common_class = data_train[outcome].value_counts().idxmax()

## training set baseline accuracy
baseline_accuracy = np.sum(data_train[outcome]==most_common_class)/len(data_train)

print("Classification accuracy (training set) if we predict all loans to be fully paid: {:.3f}"
      .format(baseline_accuracy))
```


    Classification accuracy (training set) if we predict all loans to be fully paid: 0.799


Now we train our baseline models on the subset of features. For baseline models we simply use the default DecisionTreeClassifier (which uses class_weights=None), trained on max_depths from 2 to 10. 

We store various performance metrics; in addition to the accuracy score, we also store the balanced accuracy score, precision score, and confusion matrix for each model so that we can investigate beyond a simple accuracy score.



```python
data_train_baseline = data_train[top_15_features+[outcome]]
data_val_baseline = data_val[top_15_features+[outcome]]

```




```python

def compare_tree_models(data_train, data_val, outcome, class_weights=[None], max_depths=range(2,21)):
    X_train = data_train.drop(columns=outcome)
    y_train = data_train[outcome]
    X_val = data_val.drop(columns=outcome)
    y_val = data_val[outcome]

    results_list = []
    for class_weight in class_weights:
        for depth in max_depths:
            clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, class_weight=class_weight)
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)
            train_cm = confusion_matrix(y_train, y_train_pred)

            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred)
            val_cm = confusion_matrix(y_val, y_val_pred)

            results_list.append({'Depth': depth,
                                    'class_weight': class_weight,
                                    'Train Accuracy': train_accuracy,
                                    'Train Balanced Accuracy': train_balanced_accuracy,
                                    'Train Precision': train_precision,
                                    'Train CM': train_cm,
                                    'Val Accuracy': val_accuracy,
                                    'Val Balanced Accuracy': val_balanced_accuracy,
                                    'Val Precision': val_precision,
                                    'Val CM': val_cm})
            
    columns=['Depth', 'class_weight', 'Train Accuracy', 'Train Balanced Accuracy',
    'Train Precision', 'Val Accuracy', 'Val Balanced Accuracy', 'Val Precision']
        
    results_table = pd.DataFrame(results_list, columns=columns)
    return results_table, results_list

results_table, results_list = compare_tree_models(data_train_baseline, data_val_baseline, outcome='fully_paid')
results_table
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>class_weight</th>
      <th>Train Accuracy</th>
      <th>Train Balanced Accuracy</th>
      <th>Train Precision</th>
      <th>Val Accuracy</th>
      <th>Val Balanced Accuracy</th>
      <th>Val Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>None</td>
      <td>0.799313</td>
      <td>0.500000</td>
      <td>0.799313</td>
      <td>0.799356</td>
      <td>0.500000</td>
      <td>0.799356</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>None</td>
      <td>0.799313</td>
      <td>0.500000</td>
      <td>0.799313</td>
      <td>0.799356</td>
      <td>0.500000</td>
      <td>0.799356</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>None</td>
      <td>0.799487</td>
      <td>0.536238</td>
      <td>0.811405</td>
      <td>0.800828</td>
      <td>0.539709</td>
      <td>0.812626</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>None</td>
      <td>0.801163</td>
      <td>0.530861</td>
      <td>0.809520</td>
      <td>0.800276</td>
      <td>0.532327</td>
      <td>0.810081</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>None</td>
      <td>0.801561</td>
      <td>0.516372</td>
      <td>0.804640</td>
      <td>0.800276</td>
      <td>0.516537</td>
      <td>0.804748</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>None</td>
      <td>0.802409</td>
      <td>0.533414</td>
      <td>0.810372</td>
      <td>0.799632</td>
      <td>0.531752</td>
      <td>0.809895</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8</td>
      <td>None</td>
      <td>0.804892</td>
      <td>0.532469</td>
      <td>0.810003</td>
      <td>0.798529</td>
      <td>0.527458</td>
      <td>0.808444</td>
    </tr>
    <tr>
      <th>7</th>
      <td>9</td>
      <td>None</td>
      <td>0.807804</td>
      <td>0.542317</td>
      <td>0.813327</td>
      <td>0.796046</td>
      <td>0.528136</td>
      <td>0.808716</td>
    </tr>
    <tr>
      <th>8</th>
      <td>10</td>
      <td>None</td>
      <td>0.812607</td>
      <td>0.555445</td>
      <td>0.817789</td>
      <td>0.790437</td>
      <td>0.526687</td>
      <td>0.808306</td>
    </tr>
    <tr>
      <th>9</th>
      <td>11</td>
      <td>None</td>
      <td>0.818492</td>
      <td>0.579222</td>
      <td>0.826148</td>
      <td>0.786943</td>
      <td>0.532911</td>
      <td>0.810540</td>
    </tr>
    <tr>
      <th>10</th>
      <td>12</td>
      <td>None</td>
      <td>0.825798</td>
      <td>0.597175</td>
      <td>0.832476</td>
      <td>0.782805</td>
      <td>0.534099</td>
      <td>0.811045</td>
    </tr>
    <tr>
      <th>11</th>
      <td>13</td>
      <td>None</td>
      <td>0.834677</td>
      <td>0.627171</td>
      <td>0.843528</td>
      <td>0.778943</td>
      <td>0.542324</td>
      <td>0.814104</td>
    </tr>
    <tr>
      <th>12</th>
      <td>14</td>
      <td>None</td>
      <td>0.845283</td>
      <td>0.648505</td>
      <td>0.851235</td>
      <td>0.772598</td>
      <td>0.536639</td>
      <td>0.812186</td>
    </tr>
    <tr>
      <th>13</th>
      <td>15</td>
      <td>None</td>
      <td>0.856625</td>
      <td>0.676915</td>
      <td>0.861953</td>
      <td>0.767816</td>
      <td>0.542058</td>
      <td>0.814309</td>
    </tr>
    <tr>
      <th>14</th>
      <td>16</td>
      <td>None</td>
      <td>0.869509</td>
      <td>0.706900</td>
      <td>0.873438</td>
      <td>0.758989</td>
      <td>0.540312</td>
      <td>0.813896</td>
    </tr>
    <tr>
      <th>15</th>
      <td>17</td>
      <td>None</td>
      <td>0.882343</td>
      <td>0.734165</td>
      <td>0.883942</td>
      <td>0.753839</td>
      <td>0.540009</td>
      <td>0.813922</td>
    </tr>
    <tr>
      <th>16</th>
      <td>18</td>
      <td>None</td>
      <td>0.895769</td>
      <td>0.769274</td>
      <td>0.898327</td>
      <td>0.748046</td>
      <td>0.543251</td>
      <td>0.815341</td>
    </tr>
    <tr>
      <th>17</th>
      <td>19</td>
      <td>None</td>
      <td>0.908796</td>
      <td>0.797957</td>
      <td>0.910013</td>
      <td>0.741517</td>
      <td>0.540368</td>
      <td>0.814411</td>
    </tr>
    <tr>
      <th>18</th>
      <td>20</td>
      <td>None</td>
      <td>0.920578</td>
      <td>0.827289</td>
      <td>0.922590</td>
      <td>0.734069</td>
      <td>0.541717</td>
      <td>0.815169</td>
    </tr>
  </tbody>
</table>
</div>





```python

def print_confusion_matrix(confusion_matrix, class_names, figsize = (10,7), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.
    
    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix. 
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.
        
    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """
    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names, 
    )
    fig = plt.figure(figsize=figsize)
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d")
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig
```




```python
best_model_index = results_table['Val Accuracy'].idxmax()
cm = results_list[best_model_index]['Val CM']

fig = print_confusion_matrix(cm, class_names=['Charged Off', 'Fully Paid'])

```



![png](Models_files/Models_16_0.png)


Comments: Our baseline accuracy values are not impressive; the best accuracy score on the validation set is about 80%, which is the accuracy we'd achieve if we simply predicted all loans to be fully paid.

Though we have used accuracy as the scoring metric for our baseline model, we realize that this is not the most appropriate metric to use going forward. We should also consider other performance metrics, such as precision scores and balanced accuracy scores.

Going forward, we should also place more weight (i.e., by modifyinng the `class_weight` when training models) on correctly classifying loans that are charged off. There are two main reasons for this:
1. 80% of the loans in our dataset are, in fact, fully paid. We should add more weight to the charged-off loans to account for this imbalance in the loan outcome labels.
2. For the purposes of building a sound, reasonably low-risk investment strategy, we hope to minimize the particular error in which the model predicts a loan to be fully paid when it is truly charged off. We will tune the class_weight parameter in future models to select an optimal value for our purposes.


## <font color='maroon'>Feature Engineering Attempts</font>

To enhance model performance while tuning our models, , we explored generating additional features not in the raw data set to potentially catch additional relationships in the response variable. 

Our attempts included interaction variables between top features, as well as polynomial terms with degree 2. Unfortunately, we did not see any notable model performance from these changes, so they were not included in our final model. 

There are many other higher-order polynomial and interaction terms and combinations of relevant/related predictors that could also be fine-tuned into better summary variables to boost model performance. This feature engineering step is an area where there is room for substantial improvement upon our model in the future. 



```python
def sum_cols_to_new_feature(df, new_feature_name, cols_to_sum, drop=True):
    new_df = df.copy()
    new_df[new_feature_name] = 0
    for col in cols_to_sum:
        new_df[new_feature_name] = new_df[new_feature_name] + new_df[col]
    if drop:
        new_df = new_df.drop(columns=cols_to_sum)
    return new_df

def add_interactions(loandf):
    df = loandf.copy()
    df['int_rate_X_sub_grade'] = df['int_rate']*df['sub_grade']
    df['installment_X_sub_grade'] = df['installment']*df['sub_grade']
    df['installment_X_int_rate'] = df['installment']*df['int_rate']
    df['int_rate_X_sub_grade_X_installment'] = df['int_rate']*df['sub_grade']*df['installment']
    df['dti_X_sub_grade'] = df['dti']*df['sub_grade']
    df['mo_sin_old_rev_tl_op_X_sub_grade'] = df['mo_sin_old_rev_tl_op']*df['sub_grade']
    df['dti_X_mo_sin_old_rev_tl_op_X_sub_grade'] = df['dti']*df['mo_sin_old_rev_tl_op']*df['sub_grade']
    df['income_to_loan_amount'] = df['annual_inc']/df['loan_amnt']
    df['dti_X_income'] = df['dti']*df['annual_inc']
    df['dti_X_income_X_loan_amnt'] = df['annual_inc']*df['loan_amnt']*df['dti']
    df['dti_X_loan_amnt'] = df['dti']*df['loan_amnt']
    df['avg_cur_bal_X_dti'] = df['avg_cur_bal']*df['dti']
    
    return df


```




```python
outcome='fully_paid'
#clean_df = pd.read_pickle('./data/Pickle/clean_df.pkl').sample(frac=0.05, random_state=0)
clean_df = pd.read_pickle('/content/gdrive/My Drive/Lending Club Project/data/Pickle/clean_df_Mon.pkl').sample(frac=.05, random_state=0)


clean_df = add_interactions(clean_df)
clean_df = sum_cols_to_new_feature(clean_df, new_feature_name='months_since_delinq_combined',
            cols_to_sum=['mths_since_last_delinq', 'mths_since_last_major_derog','mths_since_last_record',
            'mths_since_recent_bc_dlq', 'mths_since_recent_revol_delinq'], drop=True)

clean_df = sum_cols_to_new_feature(clean_df, new_feature_name='num_bad_records',
            cols_to_sum=['num_accts_ever_120_pd','num_tl_90g_dpd_24m','pub_rec',
                         'pub_rec_bankruptcies', 'tax_liens'], drop=True)
clean_df = sum_cols_to_new_feature(clean_df, new_feature_name='combined_credit_lim',
            cols_to_sum=['tot_hi_cred_lim','total_bc_limit','total_il_high_credit_limit',
                         'total_rev_hi_lim',], drop=True)

clean_df = sum_cols_to_new_feature(clean_df, new_feature_name='num_recent_delinq',
            cols_to_sum=['delinq_2yrs', 'chargeoff_within_12_mths', 'collections_12_mths_ex_med'], drop=True)

clean_df = sum_cols_to_new_feature(clean_df, new_feature_name='num_accounts',
            cols_to_sum=['num_bc_tl', 'num_il_tl', 'num_op_rev_tl', 'num_rev_accts',
                         'num_actv_rev_tl', 'num_actv_bc_tl', 'mort_acc'], drop=True)
                         

clean_df = clean_df.drop(columns = ['issue_d', 'zip_code', 'addr_state'])
data_train, data_test = train_test_split(clean_df, test_size=.1, stratify=clean_df[outcome], random_state=99);

X_train = data_train.drop(columns=outcome)
y_train = data_train[outcome]
rf_model = RandomForestClassifier(n_estimators=50, max_depth=50).fit(X_train, y_train)
                         
importances = pd.DataFrame({'Columns':X_train.columns,'Feature_Importances':rf_model.feature_importances_})
importances = importances.sort_values(by='Feature_Importances',ascending=False)
print(importances['Columns'].values.tolist()[:15])
```


    ['dti_X_sub_grade', 'int_rate', 'int_rate_X_sub_grade_X_installment', 'int_rate_X_sub_grade', 'dti_X_mo_sin_old_rev_tl_op_X_sub_grade', 'dti', 'income_to_loan_amount', 'avg_cur_bal', 'dti_X_loan_amnt', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op_X_sub_grade', 'bc_open_to_buy', 'installment_X_sub_grade', 'sub_grade', 'revol_bal']




```python

def compare_tree_models(data_train, data_val, outcome, class_weights=[None], max_depths=range(2,21)):
    X_train = data_train.drop(columns=outcome)
    y_train = data_train[outcome]
    X_val = data_val.drop(columns=outcome)
    y_val = data_val[outcome]

    tree_results = []
    for class_weight in class_weights:
        for depth in max_depths:
            clf = DecisionTreeClassifier(criterion='gini', max_depth=depth, class_weight=class_weight)
            clf.fit(X_train, y_train)

            y_train_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)
            train_cm = confusion_matrix(y_train, y_train_pred)

            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred)
            val_cm = confusion_matrix(y_val, y_val_pred)

            tree_results.append({'Depth': depth,
                                    'class_weight': class_weight,
                                    'Train Accuracy': train_accuracy,
                                    'Train Balanced Accuracy': train_balanced_accuracy,
                                    'Train Precision': train_precision,
                                    'Train CM': train_cm,
                                    'Val Accuracy': val_accuracy,
                                    'Val Balanced Accuracy': val_balanced_accuracy,
                                    'Val Precision': val_precision,
                                    'Val CM': val_cm})
    return tree_results


results = compare_tree_models(data_train, data_test,\
            outcome='fully_paid', class_weights=[None, 'balanced', {0:5, 1:1},{0:6, 1:1}, {0:7, 1:1}, {0:8, 1:1}])



```




```python
columns=['Depth', 'class_weight', 'Train Accuracy','Train Precision', 'Val Accuracy','Val Precision']

scores_table = pd.DataFrame(results, columns=columns)

msk = scores_table['Val Precision'] >= 0.9
scores_table[msk].sort_values(by='Val Accuracy', ascending=False).head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>class_weight</th>
      <th>Train Accuracy</th>
      <th>Train Precision</th>
      <th>Val Accuracy</th>
      <th>Val Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>4</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.530919</td>
      <td>0.908322</td>
      <td>0.523170</td>
      <td>0.902159</td>
    </tr>
    <tr>
      <th>59</th>
      <td>4</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.530919</td>
      <td>0.908322</td>
      <td>0.523170</td>
      <td>0.902159</td>
    </tr>
    <tr>
      <th>62</th>
      <td>7</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.538950</td>
      <td>0.921664</td>
      <td>0.517286</td>
      <td>0.900326</td>
    </tr>
    <tr>
      <th>78</th>
      <td>4</td>
      <td>{0: 7, 1: 1}</td>
      <td>0.474292</td>
      <td>0.917505</td>
      <td>0.470394</td>
      <td>0.914966</td>
    </tr>
    <tr>
      <th>81</th>
      <td>7</td>
      <td>{0: 7, 1: 1}</td>
      <td>0.487657</td>
      <td>0.932268</td>
      <td>0.465612</td>
      <td>0.904842</td>
    </tr>
    <tr>
      <th>83</th>
      <td>9</td>
      <td>{0: 7, 1: 1}</td>
      <td>0.495504</td>
      <td>0.954543</td>
      <td>0.460280</td>
      <td>0.906520</td>
    </tr>
    <tr>
      <th>82</th>
      <td>8</td>
      <td>{0: 7, 1: 1}</td>
      <td>0.482609</td>
      <td>0.943900</td>
      <td>0.458073</td>
      <td>0.903991</td>
    </tr>
    <tr>
      <th>79</th>
      <td>5</td>
      <td>{0: 7, 1: 1}</td>
      <td>0.460804</td>
      <td>0.922437</td>
      <td>0.457705</td>
      <td>0.917415</td>
    </tr>
    <tr>
      <th>77</th>
      <td>3</td>
      <td>{0: 7, 1: 1}</td>
      <td>0.458843</td>
      <td>0.918520</td>
      <td>0.455866</td>
      <td>0.914423</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3</td>
      <td>{0: 8, 1: 1}</td>
      <td>0.458843</td>
      <td>0.918520</td>
      <td>0.455866</td>
      <td>0.914423</td>
    </tr>
    <tr>
      <th>102</th>
      <td>9</td>
      <td>{0: 8, 1: 1}</td>
      <td>0.486145</td>
      <td>0.956453</td>
      <td>0.449430</td>
      <td>0.904676</td>
    </tr>
    <tr>
      <th>101</th>
      <td>8</td>
      <td>{0: 8, 1: 1}</td>
      <td>0.470062</td>
      <td>0.946868</td>
      <td>0.448327</td>
      <td>0.908759</td>
    </tr>
    <tr>
      <th>80</th>
      <td>6</td>
      <td>{0: 7, 1: 1}</td>
      <td>0.451036</td>
      <td>0.930842</td>
      <td>0.445200</td>
      <td>0.920203</td>
    </tr>
    <tr>
      <th>99</th>
      <td>6</td>
      <td>{0: 8, 1: 1}</td>
      <td>0.451956</td>
      <td>0.931425</td>
      <td>0.443545</td>
      <td>0.918679</td>
    </tr>
    <tr>
      <th>100</th>
      <td>7</td>
      <td>{0: 8, 1: 1}</td>
      <td>0.463461</td>
      <td>0.938127</td>
      <td>0.442810</td>
      <td>0.911654</td>
    </tr>
    <tr>
      <th>97</th>
      <td>4</td>
      <td>{0: 8, 1: 1}</td>
      <td>0.412556</td>
      <td>0.930116</td>
      <td>0.416146</td>
      <td>0.939144</td>
    </tr>
    <tr>
      <th>98</th>
      <td>5</td>
      <td>{0: 8, 1: 1}</td>
      <td>0.401459</td>
      <td>0.936237</td>
      <td>0.405112</td>
      <td>0.943955</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2</td>
      <td>{0: 8, 1: 1}</td>
      <td>0.336453</td>
      <td>0.940714</td>
      <td>0.342221</td>
      <td>0.947491</td>
    </tr>
  </tbody>
</table>
</div>





```python

clean_df = pd.read_pickle('/content/gdrive/My Drive/Lending Club Project/data/Pickle/clean_df_Mon.pkl').sample(frac=.05, random_state=0)
clean_df = clean_df.drop(columns = ['issue_d', 'zip_code', 'addr_state'])
data_train, data_test = train_test_split(clean_df, test_size=.1, stratify=clean_df[outcome], random_state=99);

X_train = data_train.drop(columns=outcome)
y_train = data_train[outcome]

results = compare_tree_models(data_train, data_test,\
            outcome='fully_paid', class_weights=[None, 'balanced', {0:5, 1:1},{0:6, 1:1}, {0:7, 1:1}, {0:8, 1:1}])



```




```python
columns=['Depth', 'class_weight', 'Train Accuracy','Train Precision', 'Val Accuracy','Val Precision']

scores_table = pd.DataFrame(results, columns=columns)

msk = scores_table['Val Precision'] >= 0.9
scores_table[msk].sort_values(by='Val Accuracy', ascending=False).head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>class_weight</th>
      <th>Train Accuracy</th>
      <th>Train Precision</th>
      <th>Val Accuracy</th>
      <th>Val Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>41</th>
      <td>5</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.550170</td>
      <td>0.905982</td>
      <td>0.546157</td>
      <td>0.900171</td>
    </tr>
    <tr>
      <th>42</th>
      <td>6</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.549413</td>
      <td>0.910787</td>
      <td>0.541192</td>
      <td>0.901085</td>
    </tr>
    <tr>
      <th>62</th>
      <td>7</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.530613</td>
      <td>0.922110</td>
      <td>0.521883</td>
      <td>0.900735</td>
    </tr>
    <tr>
      <th>61</th>
      <td>6</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.525259</td>
      <td>0.916513</td>
      <td>0.520780</td>
      <td>0.905317</td>
    </tr>
    <tr>
      <th>60</th>
      <td>5</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.519434</td>
      <td>0.912385</td>
      <td>0.516182</td>
      <td>0.905732</td>
    </tr>
  </tbody>
</table>
</div>





```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

X_train = data_train.drop(columns=outcome)
y_train = data_train[outcome]
X_val = data_test.drop(columns=outcome)
y_val = data_test[outcome]

scores = []
for i in range(2,11):
    print(i)
    clf = make_pipeline(PolynomialFeatures(degree=2, include_bias=False),
                            DecisionTreeClassifier(criterion='gini', max_depth=i,
                            class_weight={0:5, 1:1}))

    clf.fit(X_train, y_train)

    y_train_pred = clf.predict(X_train)
    y_val_pred = clf.predict(X_val)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_cm = confusion_matrix(y_train, y_train_pred)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_cm = confusion_matrix(y_val, y_val_pred)

    scores.append({'Depth': i,
                            'Train Accuracy': train_accuracy,
                            'Train Balanced Accuracy': train_balanced_accuracy,
                            'Train Precision': train_precision,
                            'Val Accuracy': val_accuracy,
                            'Val Balanced Accuracy': val_balanced_accuracy,
                            'Val Precision': val_precision,})

    clear_output()
        
columns=['Depth', 'Train Accuracy','Train Precision', 'Val Accuracy','Val Precision']
scores_table = pd.DataFrame(scores, columns=columns)
msk = scores_table['Val Precision'] >= 0.9
scores_table[msk].sort_values(by='Val Accuracy', ascending=False).head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>Train Accuracy</th>
      <th>Train Precision</th>
      <th>Val Accuracy</th>
      <th>Val Precision</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>





```python
scores_table.sort_values(by='Val Precision', ascending=False).head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Depth</th>
      <th>Train Accuracy</th>
      <th>Train Precision</th>
      <th>Val Accuracy</th>
      <th>Val Precision</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>0.520517</td>
      <td>0.905729</td>
      <td>0.514160</td>
      <td>0.899906</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>0.520517</td>
      <td>0.905729</td>
      <td>0.514160</td>
      <td>0.899906</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6</td>
      <td>0.590060</td>
      <td>0.907898</td>
      <td>0.584038</td>
      <td>0.894917</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>0.579168</td>
      <td>0.903548</td>
      <td>0.574476</td>
      <td>0.894328</td>
    </tr>
    <tr>
      <th>5</th>
      <td>7</td>
      <td>0.604753</td>
      <td>0.911508</td>
      <td>0.587900</td>
      <td>0.890167</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='maroon'>Decision Tree Model Tuning/Comparison</font>




```python
clean_df = pd.read_pickle('/content/gdrive/My Drive/Lending Club Project/data/Pickle/clean_df_Mon.pkl').sample(frac=.05, random_state=0)

outcome='fully_paid'

data_train, data_test = train_test_split(clean_df, test_size=.1, stratify=clean_df[outcome], random_state=99);
print(data_train.shape, data_test.shape)
data_train, data_val = train_test_split(data_train, test_size=.2, stratify=data_train[outcome], random_state=99)
print(data_train.shape, data_val.shape)

X_train = data_train.drop(columns=['issue_d', 'zip_code', 'addr_state', outcome])
y_train = data_train[outcome]

importances = ['int_rate', 'sub_grade', 'dti', 'installment', 'avg_cur_bal', 'mo_sin_old_rev_tl_op', 'bc_open_to_buy', 'credit_line_age', 'tot_hi_cred_lim', 'annual_inc', 'revol_util', 'bc_util', 'mo_sin_old_il_acct', 'revol_bal', 'total_rev_hi_lim', 'total_bc_limit', 'tot_cur_bal', 'total_bal_ex_mort', 'loan_amnt', 'total_il_high_credit_limit', 'mths_since_recent_bc', 'total_acc', 'mo_sin_rcnt_rev_tl_op', 'num_rev_accts', 'num_il_tl', 'grade', 'mths_since_recent_inq', 'mo_sin_rcnt_tl', 'num_bc_tl', 'acc_open_past_24mths', 'open_acc', 'num_sats', 'pct_tl_nvr_dlq', 'num_op_rev_tl', 'mths_since_last_delinq', 'percent_bc_gt_75', 'term_ 60 months', 'num_actv_rev_tl', 'num_rev_tl_bal_gt_0', 'num_bc_sats', 'num_actv_bc_tl', 'num_tl_op_past_12m', 'mths_since_recent_revol_delinq', 'mort_acc', 'mths_since_last_major_derog', 'mths_since_recent_bc_dlq', 'tot_coll_amt', 'mths_since_last_record', 'inq_last_6mths', 'num_accts_ever_120_pd', 'delinq_2yrs', 'pub_rec', 'verification_status_Verified', 'verification_status_Source Verified', 'emp_length_10+ years', 'purpose_debt_consolidation', 'emp_length_5-9 years', 'emp_length_2-4 years', 'home_ownership_RENT', 'purpose_credit_card', 'pub_rec_bankruptcies', 'home_ownership_MORTGAGE', 'home_ownership_OWN', 'num_tl_90g_dpd_24m', 'tax_liens', 'purpose_other', 'purpose_home_improvement', 'collections_12_mths_ex_med', 'purpose_major_purchase', 'purpose_small_business', 'purpose_medical', 'application_type_Joint App', 'purpose_moving', 'chargeoff_within_12_mths', 'purpose_vacation', 'delinq_amnt', 'purpose_house', 'acc_now_delinq', 'purpose_renewable_energy', 'purpose_wedding', 'home_ownership_OTHER', 'home_ownership_NONE', 'purpose_educational']
```


    (48934, 87) (5438, 87)
    (39147, 87) (9787, 87)




```python
scores = []
features = []
for column in importances:
    features.append(column)
    print(features)
    clear_output()
    print(features)

    print(len(features))

    X_train = data_train[features]
    y_train = data_train[outcome]

    X_val = data_val[features]
    y_val = data_val[outcome]  

    X_test = data_test[features]
    y_test = data_test[outcome]  
    

    for depth in range(3,10):
      for weight in range (3, 8):
        rf_tmp = DecisionTreeClassifier(max_depth=depth, class_weight={0:weight, 1:1}).fit(X_train, y_train)

        y_train_pred = rf_tmp.predict(X_train)
        y_val_pred = rf_tmp.predict(X_val)
        y_test_pred = rf_tmp.predict(X_test)    

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        
        test_accuracy = accuracy_score(y_test,y_test_pred)  
        test_balanced_accuracy = balanced_accuracy_score(y_test,y_test_pred)
        test_precision = precision_score(y_test,y_test_pred)  
        
        val_accuracy = accuracy_score(y_val,y_val_pred)   
        val_balanced_accuracy = balanced_accuracy_score(y_val,y_val_pred)
        val_precision = precision_score(y_val,y_val_pred) 

        tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
        train_approved = tp
        
        tn, fp, fn, tp = confusion_matrix(y_test,y_test_pred).ravel()
        test_approved = tp
        
        tn, fp, fn, tp = confusion_matrix(y_val,y_val_pred).ravel()  
        val_approved = tp

        scores.append({'Top_N_Features': len(features),
                       'Depth': depth,
                       'Weight': rf_tmp.get_params()['class_weight'],
                       'Test Accuracy': test_accuracy,
                       'Test Balanced Accuracy': test_balanced_accuracy,
                       'Test Precision': test_precision,
                       'Test Fully Paid Loans': test_approved,
                       
                       'Val Accuracy': val_accuracy,
                       'Val Balanced Accuracy': val_balanced_accuracy,
                       'Val Precision': val_precision,
                       'Val Fully Paid Loans': val_approved,
                       
                       'Train Accuracy': train_accuracy,
                       'Train Balanced Accuracy': train_balanced_accuracy,
                       'Train Precision': train_precision,
                       'Train Fully Paid Loans': train_approved,
                     
                      })
table_col_names = ['Top_N_Features','Depth','Weight', 'Test Accuracy', 'Test Balanced Accuracy','Test Precision','Test Fully Paid Loans',
'Val Accuracy','Val Balanced Accuracy','Val Precision','Val Fully Paid Loans','Train Accuracy','Train Balanced Accuracy',
'Train Precision','Train Fully Paid Loans']

tree_scores = pd.DataFrame(scores)
tree_scores = tree_scores[table_col_names]
```




```python
path = '/content/gdrive/My Drive/Lending Club Project/data/Victor/'
#tree_scores.to_csv(path+'tree_scores.csv',index=False)
tree_scores = pd.read_csv(path+'tree_scores.csv')
tree_scores[(tree_scores['Val Precision']>=0.9)&(tree_scores['Test Precision']>=0.9)].sort_values('Val Fully Paid Loans',ascending=False).head()

```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Top_N_Features</th>
      <th>Depth</th>
      <th>Weight</th>
      <th>Test Accuracy</th>
      <th>Test Balanced Accuracy</th>
      <th>Test Precision</th>
      <th>Test Fully Paid Loans</th>
      <th>Val Accuracy</th>
      <th>Val Balanced Accuracy</th>
      <th>Val Precision</th>
      <th>Val Fully Paid Loans</th>
      <th>Train Accuracy</th>
      <th>Train Balanced Accuracy</th>
      <th>Train Precision</th>
      <th>Train Fully Paid Loans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>467</th>
      <td>14</td>
      <td>5</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.533468</td>
      <td>0.631052</td>
      <td>0.90031</td>
      <td>2032</td>
      <td>0.538469</td>
      <td>0.634381</td>
      <td>0.901655</td>
      <td>3704</td>
      <td>0.541344</td>
      <td>0.641536</td>
      <td>0.907904</td>
      <td>14817</td>
    </tr>
    <tr>
      <th>432</th>
      <td>13</td>
      <td>5</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.533468</td>
      <td>0.631052</td>
      <td>0.90031</td>
      <td>2032</td>
      <td>0.538469</td>
      <td>0.634381</td>
      <td>0.901655</td>
      <td>3704</td>
      <td>0.541344</td>
      <td>0.641536</td>
      <td>0.907904</td>
      <td>14817</td>
    </tr>
    <tr>
      <th>572</th>
      <td>17</td>
      <td>5</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.533468</td>
      <td>0.631052</td>
      <td>0.90031</td>
      <td>2032</td>
      <td>0.538469</td>
      <td>0.634381</td>
      <td>0.901655</td>
      <td>3704</td>
      <td>0.541344</td>
      <td>0.641536</td>
      <td>0.907904</td>
      <td>14817</td>
    </tr>
    <tr>
      <th>502</th>
      <td>15</td>
      <td>5</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.533468</td>
      <td>0.631052</td>
      <td>0.90031</td>
      <td>2032</td>
      <td>0.538469</td>
      <td>0.634381</td>
      <td>0.901655</td>
      <td>3704</td>
      <td>0.541344</td>
      <td>0.641536</td>
      <td>0.907904</td>
      <td>14817</td>
    </tr>
    <tr>
      <th>607</th>
      <td>18</td>
      <td>5</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.533468</td>
      <td>0.631052</td>
      <td>0.90031</td>
      <td>2032</td>
      <td>0.538469</td>
      <td>0.634381</td>
      <td>0.901655</td>
      <td>3704</td>
      <td>0.541344</td>
      <td>0.641536</td>
      <td>0.907904</td>
      <td>14817</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='maroon'>Random Forest Model Tuning/Comparison</font>




```python
rf_scores = []
features = []
for column in importances[:20]:
    features.append(column)
    clear_output()
    #print(len(features))
    print(features)


    X_train = data_train[features]
    print(X_train.shape)
    y_train = data_train[outcome]

    X_val = data_val[features]
    y_val = data_val[outcome]  

    X_test = data_test[features]
    y_test = data_test[outcome]  
    

    for depth in range(3,10):
      for weight in range (3, 8):
        #rf_tmp = DecisionTreeClassifier(max_depth=depth, class_weight={0:weight, 1:1}).fit(X_train, y_train)
        rf_tmp = RandomForestClassifier(n_estimators=20, max_depth=depth, class_weight={0:weight, 1:1}).fit(X_train, y_train)
        print(X_train.shape, X_train.columns)
        
        y_train_pred = rf_tmp.predict(X_train)
        y_val_pred = rf_tmp.predict(X_val)
        y_test_pred = rf_tmp.predict(X_test)    

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_balanced_accuracy = balanced_accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        
        test_accuracy = accuracy_score(y_test,y_test_pred)  
        test_balanced_accuracy = balanced_accuracy_score(y_test,y_test_pred)
        test_precision = precision_score(y_test,y_test_pred)  
        
        val_accuracy = accuracy_score(y_val,y_val_pred)   
        val_balanced_accuracy = balanced_accuracy_score(y_val,y_val_pred)
        val_precision = precision_score(y_val,y_val_pred) 

        tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
        train_approved = tp
        
        tn, fp, fn, tp = confusion_matrix(y_test,y_test_pred).ravel()
        test_approved = tp
        
        tn, fp, fn, tp = confusion_matrix(y_val,y_val_pred).ravel()  
        val_approved = tp

        rf_scores.append({'Top_N_Features': len(features),
                       'Depth': depth,
                       'Weight': rf_tmp.get_params()['class_weight'],
                       'Test Accuracy': test_accuracy,
                       'Test Balanced Accuracy': test_balanced_accuracy,
                       'Test Precision': test_precision,
                       'Test Fully Paid Loans': test_approved,
                       
                       'Val Accuracy': val_accuracy,
                       'Val Balanced Accuracy': val_balanced_accuracy,
                       'Val Precision': val_precision,
                       'Val Fully Paid Loans': val_approved,
                       
                       'Train Accuracy': train_accuracy,
                       'Train Balanced Accuracy': train_balanced_accuracy,
                       'Train Precision': train_precision,
                       'Train Fully Paid Loans': train_approved,
                     
                      })
        
table_col_names = ['Top_N_Features','Depth','Weight', 'Test Accuracy', 'Test Balanced Accuracy','Test Precision','Test Fully Paid Loans',
'Val Accuracy','Val Balanced Accuracy','Val Precision','Val Fully Paid Loans','Train Accuracy','Train Balanced Accuracy',
'Train Precision','Train Fully Paid Loans']

rf_scores = pd.DataFrame(rf_scores)
rf_scores = rf_scores[table_col_names]

```




```python
path = '/content/gdrive/My Drive/Lending Club Project/data/Victor/'
#rf_scores.to_csv(path+'rf_scores.csv',index=False)
rf_scores = pd.read_csv(path+'rf_scores.csv')
rf_scores[(rf_scores['Val Precision']>=0.9)&(rf_scores['Test Precision']>=0.9)].sort_values('Val Fully Paid Loans',ascending=False).head()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Top_N_Features</th>
      <th>Depth</th>
      <th>Weight</th>
      <th>Test Accuracy</th>
      <th>Test Balanced Accuracy</th>
      <th>Test Precision</th>
      <th>Test Fully Paid Loans</th>
      <th>Val Accuracy</th>
      <th>Val Balanced Accuracy</th>
      <th>Val Precision</th>
      <th>Val Fully Paid Loans</th>
      <th>Train Accuracy</th>
      <th>Train Balanced Accuracy</th>
      <th>Train Precision</th>
      <th>Train Fully Paid Loans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1498</th>
      <td>43</td>
      <td>8</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.549834</td>
      <td>0.637193</td>
      <td>0.900000</td>
      <td>2133</td>
      <td>0.553387</td>
      <td>0.638784</td>
      <td>0.900139</td>
      <td>3876</td>
      <td>0.577081</td>
      <td>0.679289</td>
      <td>0.930936</td>
      <td>15892</td>
    </tr>
    <tr>
      <th>1307</th>
      <td>38</td>
      <td>5</td>
      <td>{0: 5, 1: 1}</td>
      <td>0.548915</td>
      <td>0.636959</td>
      <td>0.900127</td>
      <td>2127</td>
      <td>0.552978</td>
      <td>0.639478</td>
      <td>0.900979</td>
      <td>3867</td>
      <td>0.557616</td>
      <td>0.647306</td>
      <td>0.906901</td>
      <td>15547</td>
    </tr>
    <tr>
      <th>1323</th>
      <td>38</td>
      <td>8</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.548547</td>
      <td>0.638097</td>
      <td>0.901402</td>
      <td>2121</td>
      <td>0.552774</td>
      <td>0.639350</td>
      <td>0.900932</td>
      <td>3865</td>
      <td>0.578461</td>
      <td>0.681102</td>
      <td>0.932163</td>
      <td>15926</td>
    </tr>
    <tr>
      <th>588</th>
      <td>17</td>
      <td>8</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.548731</td>
      <td>0.637870</td>
      <td>0.901104</td>
      <td>2123</td>
      <td>0.551957</td>
      <td>0.639218</td>
      <td>0.901122</td>
      <td>3855</td>
      <td>0.570925</td>
      <td>0.675008</td>
      <td>0.929487</td>
      <td>15660</td>
    </tr>
    <tr>
      <th>1638</th>
      <td>47</td>
      <td>8</td>
      <td>{0: 6, 1: 1}</td>
      <td>0.549283</td>
      <td>0.638557</td>
      <td>0.901570</td>
      <td>2125</td>
      <td>0.552570</td>
      <td>0.641310</td>
      <td>0.902954</td>
      <td>3852</td>
      <td>0.577362</td>
      <td>0.681269</td>
      <td>0.932906</td>
      <td>15865</td>
    </tr>
  </tbody>
</table>
</div>



## <font color='maroon'>Comparison of Top Decision Tree vs. Top Random Forest Models</font>




```python
clean_df = pd.read_pickle('/content/gdrive/My Drive/Lending Club Project/data/Pickle/clean_df_Mon.pkl')

print('Total Number of Rows:', '{:,}'.format(clean_df.shape[0]))
print('Total Number of Columns:', '{:,}'.format(clean_df.shape[1]))
```


    Total Number of Rows: 1,087,436
    Total Number of Columns: 87




```python
outcome='fully_paid'

data_train, data_test = train_test_split(clean_df, test_size=.1, stratify=clean_df[outcome], random_state=99);
print(data_train.shape, data_test.shape)
data_train, data_val = train_test_split(data_train, test_size=.2, stratify=data_train[outcome], random_state=99)
print(data_train.shape, data_val.shape)
```


    (978692, 87) (108744, 87)
    (782953, 87) (195739, 87)




```python
importances = ['int_rate', 'sub_grade', 'dti', 'installment', 'avg_cur_bal', 'mo_sin_old_rev_tl_op', 'bc_open_to_buy', 'credit_line_age', 'tot_hi_cred_lim', 'annual_inc', 'revol_util', 'bc_util', 'mo_sin_old_il_acct', 'revol_bal', 'total_rev_hi_lim', 'total_bc_limit', 'tot_cur_bal', 'total_bal_ex_mort', 'loan_amnt', 'total_il_high_credit_limit', 'mths_since_recent_bc', 'total_acc', 'mo_sin_rcnt_rev_tl_op', 'num_rev_accts', 'num_il_tl', 'grade', 'mths_since_recent_inq', 'mo_sin_rcnt_tl', 'num_bc_tl', 'acc_open_past_24mths', 'open_acc', 'num_sats', 'pct_tl_nvr_dlq', 'num_op_rev_tl', 'mths_since_last_delinq', 'percent_bc_gt_75', 'term_ 60 months', 'num_actv_rev_tl', 'num_rev_tl_bal_gt_0', 'num_bc_sats', 'num_actv_bc_tl', 'num_tl_op_past_12m', 'mths_since_recent_revol_delinq', 'mort_acc', 'mths_since_last_major_derog', 'mths_since_recent_bc_dlq', 'tot_coll_amt', 'mths_since_last_record', 'inq_last_6mths', 'num_accts_ever_120_pd', 'delinq_2yrs', 'pub_rec', 'verification_status_Verified', 'verification_status_Source Verified', 'emp_length_10+ years', 'purpose_debt_consolidation', 'emp_length_5-9 years', 'emp_length_2-4 years', 'home_ownership_RENT', 'purpose_credit_card', 'pub_rec_bankruptcies', 'home_ownership_MORTGAGE', 'home_ownership_OWN', 'num_tl_90g_dpd_24m', 'tax_liens', 'purpose_other', 'purpose_home_improvement', 'collections_12_mths_ex_med', 'purpose_major_purchase', 'purpose_small_business', 'purpose_medical', 'application_type_Joint App', 'purpose_moving', 'chargeoff_within_12_mths', 'purpose_vacation', 'delinq_amnt', 'purpose_house', 'acc_now_delinq', 'purpose_renewable_energy', 'purpose_wedding', 'home_ownership_OTHER', 'home_ownership_NONE', 'purpose_educational']
```




```python
features = importances[0:43]
depth = 8
weight = 6

X_train = data_train[features]
y_train = data_train[outcome]

X_val = data_val[features]
y_val = data_val[outcome]  

X_test = data_test[features]
y_test = data_test[outcome]  
    
rf_model = RandomForestClassifier(n_estimators=50, max_depth=depth, class_weight={0:weight, 1:1}).fit(X_train, y_train)


rf_model = RandomForestClassifier(n_estimators=50, max_depth=8, class_weight={0:6, 1:1}).fit(X_train, y_train)

```




```python
data_test['RF_Model_Prediction'] = rf_model.predict(X_test) 
data_test['RF_Probability_Fully_Paid'] = rf_model.predict_proba(X_test)[:,1]

```




```python
features = importances[0:13]
depth = 5
weight = 5

X_train = data_train[features]
y_train = data_train[outcome]

X_val = data_val[features]
y_val = data_val[outcome]  

X_test = data_test[features]
y_test = data_test[outcome]  
    
rf_model = DecisionTreeClassifier(max_depth=depth, class_weight={0:weight, 1:1}).fit(X_train, y_train)

```




```python
data_test['Tree_Model_Prediction'] = rf_model.predict(X_test) 
data_test['Tree_Probability_Fully_Paid'] = rf_model.predict_proba(X_test)[:,1]

```




```python
data_test[['RF_Model_Prediction','Tree_Model_Prediction','RF_Probability_Fully_Paid','Tree_Probability_Fully_Paid']].describe()
```





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RF_Model_Prediction</th>
      <th>Tree_Model_Prediction</th>
      <th>RF_Probability_Fully_Paid</th>
      <th>Tree_Probability_Fully_Paid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>108744.000000</td>
      <td>108744.000000</td>
      <td>108744.000000</td>
      <td>108744.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.366908</td>
      <td>0.436806</td>
      <td>0.447268</td>
      <td>0.484344</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.481963</td>
      <td>0.495993</td>
      <td>0.167528</td>
      <td>0.178238</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.124550</td>
      <td>0.187470</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.318232</td>
      <td>0.357840</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.421940</td>
      <td>0.455652</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.563963</td>
      <td>0.616304</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.888068</td>
      <td>0.864915</td>
    </tr>
  </tbody>
</table>
</div>



With similar total funded fully paid loans between the best Decision Tree and Random Forest, we need to look at the probability distributions. Because it is unrealistic to expect investors to invest in all loans we recommend, we will rank the loans by order of probability. As such, the performance of the models at predicting loans near 1.0 matters more than the loans near 0.5 as those will be fulfilled first. 



```python
fig, ax = plt.subplots(figsize=(16,10))

sns.distplot(data_test[data_test['fully_paid']==1]['RF_Probability_Fully_Paid'],color='green',rug=False,label='Fully Paid')
sns.distplot(data_test[data_test['fully_paid']==0]['RF_Probability_Fully_Paid'],color='red',rug=False,label='Charged Off')

ax.set_title('Density Distribution of Random Forest Probability')
ax.set_xlabel('Probability of Fully Paid')
ax.set_ylabel('Density')
plt.legend(loc='upper left')

sns.despine()
```



![png](Models_files/Models_44_0.png)




```python
fig, ax = plt.subplots(figsize=(16,10))

sns.distplot(data_test[data_test['fully_paid']==1]['Tree_Probability_Fully_Paid'],color='green',rug=False,label='Fully Paid')
sns.distplot(data_test[data_test['fully_paid']==0]['Tree_Probability_Fully_Paid'],color='red',rug=False,label='Charged Off')


ax.set_title('Density Distribution of Decision Tree Probability')
ax.set_xlabel('Probability of Fully Paid')
ax.set_ylabel('Density')
plt.legend(loc='upper left')

sns.despine()
```



![png](Models_files/Models_45_0.png)




```python
from sklearn.calibration import calibration_curve
rf_positive_frac, rf_mean_score = calibration_curve(y_test, data_test['RF_Probability_Fully_Paid'].values, n_bins=25)
tree_positive_frac, tree_mean_score = calibration_curve(y_test, data_test['Tree_Probability_Fully_Paid'].values, n_bins=25)

fig, ax = plt.subplots(figsize=(16,10))

ax.plot(rf_mean_score,rf_positive_frac, color='green', label='Random Forest')
ax.plot(tree_mean_score,tree_positive_frac, color='blue', label='Decision Tree')
ax.plot([0, 1], [0, 1], color='grey', label='Calibrated')

ax.set_title('Calibration Curve Comparison')
ax.set_xlabel('Probability of Fully Paid')
ax.set_ylabel('Proportion of Fully Paid')
ax.set_xlim([.1, .9])
plt.legend(loc='lower right')

sns.despine()
```



![png](Models_files/Models_46_0.png)


Because we chose models with a high base precision, the calibration curve is not suprising. 
<br>Our use case will only focus on the upper end of the probability scores where an approved charged off loan is the worst misclasification so it is not a big issue that the probabilities are not 1:1 to the proportion of fully paid loans. What we see is that our chosen Random Forest model has higher proportion of fully paid loans in comparison to Decision Tree at high probability values. While the Decision Tree is more interpretable, Random Forest is a good balance between performance and interpretability.
