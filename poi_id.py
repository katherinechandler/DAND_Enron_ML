#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from tester import test_classifier


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'to_messages',
                 'deferral_payments',
                 'expenses',
                 'deferred_income',
                 'long_term_incentive',
                 'shared_receipt_with_poi',
                 'from_messages',
                 'other',
                 'bonus',
                 'total_stock_value',
                 'from_poi_to_this_person',
                 'from_this_person_to_poi',
                 'restricted_stock',
                 'salary',
                 'total_payments',
                 'poi_email_to_fraction',
                 'exercised_stock_options']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers

data_dict.pop( 'TOTAL', 0 )

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

# I began by calculating compensation ratios. This didn't yeild very good models so I eventually engineered
# rations based on communication with POIs

# calculated fraction of compensation derived from individual feature
def calc_fraction_comp(feature,feature_total):
    calc_fraction_list=[]

    for i in my_dataset:
        if my_dataset[i][feature]=="NaN" or my_dataset[i][feature_total]=="NaN":
            my_dataset[i]["fraction_"+feature] = 0.
        elif my_dataset[i][feature]>=0:
            my_dataset[i]["fraction_"+feature] = float(my_dataset[i][feature])/float(my_dataset[i][feature_total])

def validate_feature(person, feature):
    if person[feature] == 'NaN':
        return False
    
    return True

# calculates fraction of email communication with POI                  
def add_poi_message_features(data):
    features_to_check = ['to_messages', 'from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi']
    for val in data_dict:
        individual = data_dict[val]
        features_valid = True
        for feature in features_to_check:
            if not validate_feature(individual, feature):
                individual['poi_email_total_fraction'] = 'NaN'
                individual['poi_email_to_fraction'] = 'NaN'
                individual['poi_email_from_fraction'] = 'NaN'
                features_valid = False

        if features_valid:
            total_emails = individual['to_messages'] + individual['from_messages']
            total_poi_messages = individual['from_poi_to_this_person'] + individual['from_this_person_to_poi']
            #individual['poi_email_total_fraction'] = float(total_poi_messages) / total_emails
            individual['poi_email_to_fraction'] = float(individual['from_this_person_to_poi']) / individual['from_messages']
            #individual['poi_email_from_fraction'] = float(individual['from_poi_to_this_person']) / total_emails

def add_to_feature_list(new_features, feature_list):
    for feat in new_features:
        if feat not in feature_list:
            feature_list.append(feat)
    
    return feature_list


## add'cash' equivalent fractions
calc_fraction_comp("salary","total_payments")
calc_fraction_comp("bonus","total_payments")
calc_fraction_comp("expenses","total_payments")
calc_fraction_comp("other","total_payments")
calc_fraction_comp("loan_advances","total_payments")

## add 'stock' equivalent fractions
calc_fraction_comp("exercised_stock_options","total_stock_value")
calc_fraction_comp("restricted_stock","total_stock_value")

new_features = ['poi_email_total_fraction', 'poi_email_to_fraction', 'poi_email_from_fraction']

add_poi_message_features(my_dataset)
#features_list = add_to_feature_list(new_features, features_list)


### Extract features and labels, make testing and training sets
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# let's test a handful of classifiers to see what works best; I'll use default settings for now

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

class_names = ["Nearest Neighbors", "Decision Tree", "Random Forest", "AdaBoost", "Extra Trees"]

classifiers = [
    KNeighborsClassifier(),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    ExtraTreesClassifier()]


# iterate over classifiers, commented out for brevity
#for name, clf in zip(class_names, classifiers):
#        clf.fit(features_train,labels_train)
#        scores = clf.score(features_test,labels_test)
#        print " "
#        print "Classifier and default parameters:"
#        test_classifier(clf, my_dataset, features_list, 50)
#        print "Accuracy: %0.2f" % (scores.mean())
#        print "_________________________________________________________________"
#        print " "


# Final classifier using tuned parameters outlined below.
clf = AdaBoostClassifier(algorithm='SAMME', base_estimator=None, learning_rate=1.5,
                         n_estimators=5, random_state=None)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

import warnings
warnings.filterwarnings('ignore')
import sklearn
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score, recall_score, f1_score

def model_scorer(clf, feat, labels):
    average_type = 'micro'
    pred = clf.predict(feat)
    prec_value = precision_score(labels, pred, average = average_type)
    rec_value = recall_score(labels, pred, average = average_type)
    f1_val = f1_score(labels, pred, average = 'macro')
    
    if prec_value < 0.3 or rec_value < 0.3:
        return 0.0
    else:
        return f1_val

def scoring(estimator, features_test, labels_test):
     labels_pred = estimator.predict(features_test)
     p = sklearn.metrics.precision_score(labels_test, labels_pred, average='micro')
     r = sklearn.metrics.recall_score(labels_test, labels_pred, average='micro')
     if p > 0.3 and r > 0.3:
            return sklearn.metrics.f1_score(labels_test, labels_pred, average='macro')
     return 0


def tuning_classifier():

    params = parameters = {'learning_rate': [0.5,1.0,1.5,2.0],
                           'n_estimators':[5,10,50,100],
                           'algorithm':('SAMME', 'SAMME.R')}
      
    clf = GridSearchCV(AdaBoostClassifier(), params, scoring=model_scorer, cv =cv)
    clf.fit(features,labels)


    print("Best parameters set found on development set:")
    print()
    print(clf.best_estimator_)
    print()
    print("Grid scores on development set:")
    print()
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = labels_test, clf.predict(features_test)
    print(classification_report(y_true, y_pred))
    print()

#tuning_classifier()


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)