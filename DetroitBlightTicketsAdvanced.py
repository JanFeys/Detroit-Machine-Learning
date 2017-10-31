#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import validation_curve, cross_val_score
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
import matplotlib.pyplot as plt

def load_data():
    print('loading data')
    
    #read the training and test data in as dataframes
    train_df = pd.read_csv('data/train.csv', encoding='latin-1', usecols= ['ticket_id', 'ticket_issued_date', 'hearing_date', 'violation_code', 'disposition', 'fine_amount', 'discount_amount', 'judgment_amount','compliance'], dtype = {'ticket_id': np.int64, 'ticket_issued_date': str, 'hearing_date': str, 'violation_code': str, 'disposition': str, 'fine_amount': np.float64, 'discount_amount': np.float64, 'judgment_amount': np.float64,'compliance': np.float16})
    test_df = pd.read_csv('data/test.csv', encoding='latin-1', usecols= ['ticket_id', 'ticket_issued_date', 'hearing_date', 'violation_code', 'disposition', 'fine_amount', 'discount_amount', 'judgment_amount'], dtype = {'ticket_id': np.int64, 'ticket_issued_date': str, 'hearing_date': str, 'violation_code': str, 'disposition': str, 'fine_amount': np.float64, 'discount_amount': np.float64, 'judgment_amount': np.float64})

    #The following columns were not included.
    #*violator's mailing address:
    #['mailing_address_str_number', 'mailing_address_str_name', 'city', 'state','zip_code', 'non_us_str_code', 'country']
    #*specific names:
    #['agency_name', 'inspector_name', 'violator_name']
    #*description (because we have a violation code):
    #['violation_description']
    #*all standard fee information:
    #['admin_fee', 'state_fee', 'late_fee']
    #*violation zip code because it was not recorded apparently:
    #['violation zip code']
    #*compliance and payment information (present in training set only):
    #['payment_amount', 'payment_date', 'payment_status','balance_due','collection_status','compliance_detail']
    #*clean_up_cost because it is always NaN in the training set, but not necessarily in test set:
    #['clean_up_cost']
    #*grafitti_status because it is always 0 in the training set, but not necessarily in test set:
    #['grafitti_status']
    #*for now, also the location of the infraction:
    #['violation_street_number', 'violation_street_name']
    
    #set ticket_id as the index for both
    train_df.set_index('ticket_id', inplace=True)
    test_df.set_index('ticket_id', inplace=True)
    
    return train_df, test_df

def process_data(train_df,test_df):
    print('processing data')
    
    #drop entries for which compliance is "not responsible", i.e. those valued NaN instead of binary
    train_df.dropna(subset=['compliance'], inplace=True)
    
    #make compliance column integer
    train_df['compliance'] = train_df['compliance'].apply(np.int64)
    
    #there is one NaN entry in training set's fine_amount, we drop that entry
    train_df.dropna(subset=['fine_amount'], inplace=True)

    #convert date columns into four new columns: year, month, day, dayofweek
    for col in ['ticket_issued_date', 'hearing_date']:
        day_time = pd.to_datetime(train_df[col])
        train_df.drop(col, axis=1, inplace=True)
        train_df[col[:-4]+'month'] = np.array(day_time.dt.month)
        train_df[col[:-4]+'year'] = np.array(day_time.dt.year)
        train_df[col[:-4]+'day'] = np.array(day_time.dt.day)
        train_df[col[:-4]+'dayofweek'] = np.array(day_time.dt.dayofweek)

        day_time = pd.to_datetime(test_df[col])
        test_df.drop(col, axis=1, inplace=True)
        test_df[col[:-4]+'month'] = np.array(day_time.dt.month)
        test_df[col[:-4]+'year'] = np.array(day_time.dt.year)
        test_df[col[:-4]+'day'] = np.array(day_time.dt.day)
        test_df[col[:-4]+'dayofweek'] = np.array(day_time.dt.dayofweek)

    #hearing_date may be NaN, we will replace those NaNs by -1
    cols = ['hearing_month','hearing_day','hearing_year','hearing_dayofweek']
    for col in cols:
        train_df[col].fillna(-1, inplace=True) #dayofweek can be 0, so setting to 0 is not a good idea
        test_df[col].fillna(-1, inplace=True)
    train_df[cols] = train_df[cols].applymap(np.int64)
    test_df[cols] = test_df[cols].applymap(np.int64)

    #some tools:
    #print(list(train_df.columns.values))
    #print(train_df.shape)
    #print(train_df.dtypes)
    #print(train_df.fine_amount.value_counts(dropna=False))
    #print(test_df.isnull().any())
        
    #select the columns that are not float or int so we can make those features categorical
    catcols = list(test_df.select_dtypes(exclude=['float', 'int']).columns)
    concat_df = pd.concat((train_df[catcols], test_df[catcols]), axis=0, verify_integrity=True)

    #clean up the violation codes: remove characters after space and bracket, remove codes without dash
    concat_df['violation_code'] = concat_df['violation_code'].apply(lambda x: x.split(' ')[0])
    concat_df['violation_code'] = concat_df['violation_code'].apply(lambda x: x.split('(')[0])
    concat_df['violation_code'] = concat_df['violation_code'].apply(lambda x: x.split('/')[0])
    concat_df['violation_code'][concat_df['violation_code'].apply(lambda x: x.find('-')<=0)] = '0'
    
    #set violation codes with few appearances to zero (FOR NOW below 50)
    counts = concat_df['violation_code'].value_counts()
    concat_df['violation_code'][concat_df['violation_code'].isin(counts[counts < 50].index)] = '0'
    
    #set a small number of disposition descriptions codes with very few appearances to 'misc'
    counts = concat_df['disposition'].value_counts()
    concat_df['disposition'][concat_df['disposition'].isin(counts[counts < 10].index)] = 'Misc'

    #make the features categorical
    for col in catcols:
        dummies = pd.get_dummies(concat_df[col]) #sparse=True can be ENABLED
        concat_df[dummies.columns] = dummies
        concat_df.drop(col, axis=1, inplace=True)
        train_df.drop(col, axis=1, inplace=True)
        test_df.drop(col, axis=1, inplace=True)

    #pour back into train and test set
    train_df[concat_df.columns] = concat_df.loc[train_df.index]
    test_df[concat_df.columns] = concat_df.loc[test_df.index]
    return train_df, test_df

def learn_data(train_df,test_df):
    print('learning data')

    features = list(test_df.columns)
    response = ['compliance']
    #features = ['fine_amount', 'discount_amount', 'judgment_amount']

    X = train_df[features]
    Y = np.array(train_df[response]).ravel()
    X_test = test_df[features]
    
    #normalize the numerical columns
    numcols = list(X.select_dtypes(include=['float'],exclude=['int']).columns)
    for col in numcols:
        X_col = X[col].values.astype(float)
        X_test_col = X_test[col].values.astype(float)
        col_mean = X_col.mean()
        col_std = X_col.std()
        X[col+'_norm'] = (X_col-col_mean)*(1/col_std)
        X_test[col+'_norm'] = (X_test_col-col_mean)*(1/col_std)

    for col in numcols:
        X.drop(col, axis=1, inplace=True)
        X_test.drop(col, axis=1, inplace=True)
    
    classifiers = {"SGD": SGDClassifier(penalty='l1',max_iter=5),"SGD-Perceptron": Perceptron(max_iter=5),"Passive-Aggressive": PassiveAggressiveClassifier(loss='hinge', C=1.2,max_iter=5),"Random-Forest": RandomForestClassifier(max_depth=15)}

    for classifier_type in list(classifiers.keys()):
        print('------',classifier_type)
        clf = classifiers[classifier_type]
        scores = cross_val_score(clf, X, Y, cv=5, scoring='roc_auc')
        print(scores)

    plot_validation_curve(X,Y)
    quit()
    
    clf.fit(X, Y)
    y_pred = clf.predict_proba(X_test)
    
    result_sr = pd.Series(data=y_pred[:,1], index=np.array(test_df.index.values.tolist()), dtype="float32")
    result_sr.index.name = 'ticket_id'
    return result_sr

def plot_validation_curve(X,Y):
    param_range = np.linspace(2, 30, 15)
    print('range of parameter max_depth:',param_range)
    train_scores, test_scores = validation_curve(RandomForestClassifier(), X, Y, param_name="max_depth",
                                                 param_range=param_range,cv=5, scoring="roc_auc", n_jobs=3)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with RandomForestClassifier")
    plt.xlabel("max_depth")
    plt.ylabel("roc_auc")
    plt.ylim(0.5, 1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,train_scores_mean + train_scores_std, alpha=0.2,color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,test_scores_mean + test_scores_std, alpha=0.2,color="navy", lw=lw)
    plt.legend(loc="best")
    plt.savefig('RandomForestGrid.pdf')
    #plt.show()


def blight_model():
    train_df, test_df = load_data()
    train_df, test_df = process_data(train_df,test_df)
    learn_data(train_df, test_df)
    return None

blight_model()
