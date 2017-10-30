import pandas as pd
import numpy as np

def load_data():
    print('loading data')
    
    #read the training and test data in as dataframes
    train_df = pd.read_csv('train.csv', encoding='latin-1')[::]
    test_df = pd.read_csv('test.csv', encoding='latin-1')
    
    #set ticket_id as the index for both
    train_df.set_index('ticket_id', inplace=True)
    test_df.set_index('ticket_id', inplace=True)
    
    #drop entries for which compliance is "not responsible", i.e. those whose value is not 0 or 1 but NaN
    train_df.dropna(subset=['compliance'], inplace=True)
    
    #remove some entries that are specific
    #train_df = train_df[train_df.country=='USA'] #remove 11 entries whose mailing address is abroad
    
    #drop all info to do with violator's mailing address columns
    for col in ['mailing_address_str_number', 'mailing_address_str_name', 'city', 'state', 
                   'zip_code', 'non_us_str_code', 'country']:
        train_df.drop(col, axis=1, inplace=True)
        test_df.drop(col, axis=1, inplace=True)
        
    #drop all info to do with names of agency, inspector and violator
    for col in ['agency_name', 'inspector_name', 'violator_name']:
        train_df.drop(col, axis=1, inplace=True)
        test_df.drop(col, axis=1, inplace=True)

    #drop violation description since we have an ID
    train_df.drop('violation_description', axis=1, inplace=True)
    test_df.drop('violation_description', axis=1, inplace=True)

    #drop fee information
    for col in ['admin_fee', 'state_fee', 'late_fee']:
        train_df.drop(col, axis=1, inplace=True)
        test_df.drop(col, axis=1, inplace=True)

    #drop violation zip code since it was not recorded apparently
    train_df.drop('violation_zip_code', axis=1, inplace=True)
    test_df.drop('violation_zip_code', axis=1, inplace=True)

    #drop some compliance information (present in training only) that appears to be redundant
    for col in ['payment_amount', 'payment_date', 'payment_status','balance_due','collection_status','compliance_detail']:
        train_df.drop(col, axis=1, inplace=True)
        
    #training's clean_up_cost is always zero, but it is non-zero in test set, we will drop therefore
    train_df.drop('clean_up_cost', axis=1, inplace=True)
    test_df.drop('clean_up_cost', axis=1, inplace=True)
    
    #training's grafitti_status is always NaN, but may be non-zero in test set, we will drop therefore
    train_df.drop('grafitti_status', axis=1, inplace=True)
    test_df.drop('grafitti_status', axis=1, inplace=True)
    
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
    
    #make compliance column integer
    cols = ['compliance']
    train_df[cols] = train_df[cols].applymap(np.int64)
    
    #make other columns integer
    cols = ['violation_street_number','fine_amount','discount_amount']
    train_df[cols] = train_df[cols].applymap(np.int64)
    test_df[cols] = test_df[cols].applymap(np.int64)

    #note that issue_date is never NaN, while hearing_date may be
    #we will replace those NaNs by zero
    cols = ['hearing_month','hearing_day','hearing_year','hearing_dayofweek']
    for col in cols:
        train_df[col].fillna(-1, inplace=True) #dayofweek can be 0, so setting to 0 is not a good idea
        test_df[col].fillna(-1, inplace=True)
    train_df[cols] = train_df[cols].applymap(np.int64)
    test_df[cols] = test_df[cols].applymap(np.int64)
    
    #FOR NOW: we drop the loation of the violation
    for col in ['violation_street_number', 'violation_street_name']:
        train_df.drop(col, axis=1, inplace=True)
        test_df.drop(col, axis=1, inplace=True)
    
    ##some diagnostics:
     #output all columns
    #print(list(train_df.columns.values))
    # #print datatypes for the columns
    #for col in train_df.columns:
    #    print(train_df[col].dtype, col)
     #count and print the occurrences of number in a column
    #vc = train_df.disposition.value_counts(dropna=False)
    #print(vc)
     #print out columns and shape of dataframe for info
    #print(list(train_df.columns.values))
     #print shape of the data
    #print(train_df.shape)
    #print(test_df.shape)
    return train_df, test_df

def process_data(train_df,test_df):
    print('processing data')

    #select the columns that are not float or int so we can make those features categorical
    cols = test_df.select_dtypes(exclude=['float', 'int']).columns
    concat_df = pd.concat((train_df[cols], test_df[cols]), axis=0, verify_integrity=True)
    
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
    for col in cols:
        dummies = pd.get_dummies(concat_df[col],sparse=True)
        concat_df[dummies.columns] = dummies
        concat_df.drop(col, axis=1, inplace=True)
        train_df.drop(col, axis=1, inplace=True)
        test_df.drop(col, axis=1, inplace=True)

    #pour back into train and test set
    train_df[concat_df.columns] = concat_df.loc[train_df.index]
    test_df[concat_df.columns] = concat_df.loc[test_df.index]
    
    #show columns
    #print(list(train_df.columns.values))
    
    #for col in list(test_df.columns.values):
    #    print('----',col)
    #    vc = train_df[col].value_counts(dropna=False)
    #    print(vc)
    
    return train_df, test_df

def learn_data(train_df,test_df):
    print('learning data')

    features = list(test_df.columns)
    response = ['compliance']

    X = train_df[features]
    Y = np.array(train_df[response]).ravel()

    print(list(train_df.columns.values))
    
    print('$$$$$$$$$$$$$$$')
    # Normalize
    X_norm=(X-X.min())/(X.max()-X.min())
        
    # Train classifier
    #clf = RandomForestClassifier(max_depth=25)
    #clf.fit(X, Y)
    
    # Predict
    #y_pred = np.array(clf.predict(Xtest))
    #y_pred = y_pred - y_pred.min()
    #y_pred = y_pred / y_pred.max()

    # Save
    #result_df = {"ticket_id":test_ticket_id, "compliance":y_pred}
    #result_df = pd.DataFrame(result_df, columns=["ticket_id", "compliance"])
      
    #print('$$$$$$')
    #print(result_df)
    
def blight_model():
    train_df, test_df = load_data()
    train_df, test_df = process_data(train_df,test_df)
    #learn_data(train_df, test_df)
    print('done!')
    print(dir())
    return None

blight_model()