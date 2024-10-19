#!/usr/bin/env python

import argparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import make_scorer, precision_score, f1_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV

def convert_arg(arg):
    if type(arg) is not str:
        return arg

    # Handle the case for None
    if arg == 'None':
        return None
    
    # Try converting to integer
    try:
        return int(arg)
    except (ValueError, TypeError):
        pass  # Move to the next conversion if it fails
    
    # Try converting to float
    try:
        return float(arg)
    except (ValueError, TypeError):
        pass  # Move to the next conversion if it fails
    
    # Return the argument as a string if it can't be converted
    return arg


def create_arg_parser():
    ''' Creates an argument parser '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default='train_data_it.csv', type=str,
                        help="Train file to learn from (default train_data_it.csv)")
    parser.add_argument("-d", "--test_file", default='test_data_it.csv', type=str,
                        help="Dev file to evaluate on (default test_data_it.csv)")
    parser.add_argument("-s", "--sentiment", action="store_true",
                        help="Do sentiment analysis (2-class problem)")
    parser.add_argument("-tf", "--tfidf", action="store_true",
                        help="Use the TF-IDF vectorizer instead of CountVectorizer")
    parser.add_argument("-cv", "--cross_validate", action="store_true",
                        help="Perform cross-validation on the training set and print the results")
    parser.add_argument("--top_features", action="store_true",
                        help="Show top features")
    parser.add_argument("-c", "--classifier", default="nb", type=str,
                        choices=["nb", "dt", "rf", "knn", "svm"], help="Classifier used for training (default nb)")
    
    # Random Forest & Decision Tree arguments
    parser.add_argument("--criterion", default="gini", type=str,
                        choices=["gini", "entropy", "log_loss"],
                        help="For DT/RF: The function to measure the quality of a split (default gini)")
    parser.add_argument("--max_depth", default=None,
                        help="For DT/RF: Maximum depth of the tree (default None)")
    parser.add_argument("--min_samples_leaf", default=1,
                        help="For DT/RF: Minimum number of samples a leaf node must possess (default 1). See official doc for more info")
    parser.add_argument("--max_leaf_nodes", default=None,
                        help="For DT/RF: Maximum number of leaf nodes the decision tree can have")
    
    # Random Forest arguments
    parser.add_argument("--n_estimators", default="100", type=int,
                        help="For RF: The number of trees in the forest")
    
    parser.add_argument("--nr_splits", default=2, type=int,
                        help="Specify number of stratified splits for cross-validation")
    
    # Add preprocessing options
    parser.add_argument("--preprocess", default=None, choices=["zscore", "pca", "anova"],
                        help="Preprocessing method to apply before classification")
    parser.add_argument("--pca_components", default=10, type=int,
                        help="Number of PCA components to keep (default 10)")
    parser.add_argument("--k_best", default=10, type=int,
                        help="Number of top features to select using ANOVA (default 10)")

    # Grid search
    parser.add_argument("-gs", "--grid_search", action="store_true",
                        help="Perform grid search to optimize hyperparameters")

    args = parser.parse_args()
    for arg in vars(args):
        value=getattr(args, arg)
        nvalue=convert_arg(value)
        setattr(args, arg, nvalue)
    
    return args

# Load your dataset
def load_data(file_path):
    data = pd.read_csv(file_path)  # assuming your data is in CSV format
    X = data.drop(columns=["IT_M_Label"])  # Features (remove 'IT_M_Label' or 'NST_M_Label' column)
    y = data["IT_M_Label"]
    return X, y

def measures(Y_test, Y_pred):
    '''
    Computes and prints the accuracy, precision, recall, F-score and the confusion matrix for the given data.
    '''
    np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
    acc = accuracy_score(Y_test, Y_pred)
    prc, rec, fsc, _ = precision_recall_fscore_support(Y_test, Y_pred, zero_division=0.0)
    wprc, wrec, wfsc, _ = precision_recall_fscore_support(Y_test, Y_pred, average="weighted", zero_division=0.0)
    confuse = confusion_matrix(Y_test, Y_pred)

    print(f'Accuracy test: {(acc*100):.5f}%')
    print(f'Precision test: {prc} ({(wprc*100):.5f}%)')
    print(f'Recall test: {rec} ({(wrec*100):.5f}%)')
    print(f'F-score test: {fsc} ({(wfsc*100):.5f}%)')
    print('Confusion matrix test:')
    print(confuse)

def stratified_cv(classifier, X_train, Y_train, nr_splits, args, random_state=42):
    '''
    Perform Stratified N-fold Cross Validation to ensure balanced class distribution across folds.
    Includes optional preprocessing (z-score, PCA, ANOVA).
    '''
    skf = StratifiedKFold(n_splits=nr_splits, shuffle=True, random_state=random_state)
    
    fold = 1
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': []}
    
    for train_index, val_index in skf.split(X_train, Y_train):
        X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
        Y_tr, Y_val = Y_train.iloc[train_index], Y_train.iloc[val_index]

        # Apply preprocessing on training set only
        if args.preprocess == 'zscore':
            # Standardize features (zero mean, unit variance)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_val = scaler.transform(X_val)
            print(f"Fold {fold}: Z-score normalization applied")

        elif args.preprocess == 'pca':
            # Apply PCA
            pca = PCA(n_components=args.pca_components)
            X_tr = pca.fit_transform(X_tr)
            X_val = pca.transform(X_val)
            print(f"Fold {fold}: PCA applied with {args.pca_components} components")

        elif args.preprocess == 'anova':
            # Feature selection using ANOVA (SelectKBest)
            selector = SelectKBest(f_classif, k=args.k_best)
            X_tr = selector.fit_transform(X_tr, Y_tr)
            X_val = selector.transform(X_val)  # Apply same top features to validation
            print(f"Fold {fold}: ANOVA feature selection applied, top {args.k_best} features selected")

        # Train the classifier on the training fold
        classifier.fit(X_tr, Y_tr)
        
        # Predict on the validation fold
        Y_pred = classifier.predict(X_val)

        # Compute performance metrics
        acc = accuracy_score(Y_val, Y_pred)
        prc, rec, fsc, _ = precision_recall_fscore_support(Y_val, Y_pred, average='weighted', zero_division=0.0)

        # Store the results for each fold
        metrics['Accuracy'].append(acc)
        metrics['Precision'].append(prc)
        metrics['Recall'].append(rec)
        metrics['F1-score'].append(fsc)

        # Print the results for the current fold
        print(f"Fold {fold}:")
        print(f"  Accuracy: {acc:.5f}")
        print(f"  Precision: {prc:.5f}")
        print(f"  Recall: {rec:.5f}")
        print(f"  F1-score: {fsc:.5f}")
        print()

        fold += 1

    # Print the average performance across folds
    print("Cross-validation results:")
    for metric in metrics:
        print(f"{metric}: {np.mean(metrics[metric]):.5f} (± {np.std(metrics[metric]):.5f})")
    print()


def get_classifier(vec, args):
    '''
    This script returns a Pipeline object including the vector of features 
    and the classifier indicated by the -c/--classifier argument,
    while also passing any specified parameters to the corresponding classifier.
    '''
    match args.classifier:
        case "dt":
            return DecisionTreeClassifier(criterion=args.criterion, max_depth=args.max_depth, 
                                            min_samples_leaf=args.min_samples_leaf,  
                                            max_leaf_nodes=args.max_leaf_nodes,
                                            random_state=42) # random_state for reproducibility!
        
        case "rf":
            return RandomForestClassifier(n_estimators=args.n_estimators,
                                            criterion=args.criterion, max_depth=args.max_depth, 
                                            min_samples_leaf=args.min_samples_leaf,  
                                            max_leaf_nodes=args.max_leaf_nodes,
                                            random_state=42)
        

def identity(inp):
    ''' Dummy function that just returns the input '''
    return inp

def get_vector(args):
    ''' Returns a vectorizer from the given arguments'''

    tokenizer=identity

    # Convert the texts to vectors
    # We use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    vec = TfidfVectorizer(preprocessor=identity, tokenizer=tokenizer, token_pattern=None) # ! to suppress warning 

    return vec

def print_top_features(classifier, vec, top_n=10):
    ''' 
    Retrieve each feature and respective count for each class, based on respective classifier and vectorizer
    '''
    feature_names = vec.get_feature_names_out()
    
    # For tree-based classifiers (DecisionTree, RandomForest)
    if hasattr(classifier.named_steps['cls'], 'feature_importances_'):
        importances = classifier.named_steps['cls'].feature_importances_
        top_features = np.argsort(importances)[-top_n:]
        print(f"Top {top_n} features:")
        for idx in top_features[::-1]:
            print(f"  {feature_names[idx]} ({importances[idx]:.4f})")
    else:
        print("This classifier does not support feature importance extraction.")


def perform_grid_search(X_train, Y_train, args):
    ''' Perform Grid Search for Decision Tree with Preprocessing options '''

    # Initialize the classifier (Decision Tree)
    model = RandomForestClassifier(random_state=42)
    
    # Define the steps in the pipeline (conditionally add preprocessing steps)
    steps = []
    
    if args.preprocess == 'zscore':
        steps.append(('scaler', StandardScaler()))
    elif args.preprocess == 'pca':
        steps.append(('pca', PCA()))
    elif args.preprocess == 'anova':
        steps.append(('anova', SelectKBest(f_classif)))
    
    # Add the classifier to the pipeline
    steps.append(('rf', model))
    
    # Create the pipeline
    pipeline = Pipeline(steps)
    
    # Set up the parameter grid for grid search
    param_grid = {
        'rf__n_estimators': [100,200],
        'rf__criterion': ['gini', 'entropy'],
        'rf__max_depth': [10, 20],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_leaf_nodes': [50, 100]
    }

    # Add grid search parameters based on preprocessing options
    if args.preprocess == 'pca':
        param_grid['pca__n_components'] = [5, 10, 15]  # Add PCA components as grid parameter
    elif args.preprocess == 'anova':
        param_grid['anova__k'] = [5, 10, 20]  # Add number of ANOVA features to select

    # Define scoring (F1-score with weighted average)
    scoring = make_scorer(f1_score, average='weighted', zero_division=0)
    
    # Perform the grid search
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, 
                               scoring=scoring, cv=StratifiedKFold(n_splits=args.nr_splits, shuffle=True, random_state=42))
    
    grid_search.fit(X_train, Y_train)
    
    # Print the best parameters and best score
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.5f}")
    
    # Print all results of the grid search
    print("\nAll results:")
    results = grid_search.cv_results_
    for i in range(len(results['mean_test_score'])):
        print(f"Mean F1-score: {results['mean_test_score'][i]:.5f} (std: {results['std_test_score'][i]:.5f}) for params: {results['params'][i]}")
    
    # Plotting the results
    plot_results(results)

    return grid_search.best_estimator_


def plot_results(grid_search_results):
    """
    Create a scatter plot showing the relationship between F1-scores and min_samples_leaf.
    """
    f1_scores = grid_search_results['mean_test_score']  # Mean F1-scores
    params = grid_search_results['params']  # Parameters used in the grid search
    
    # Extract the min_samples_leaf values
    min_samples_leaf_values = [param['rf__min_samples_leaf'] for param in params]
    
    # Create a scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(min_samples_leaf_values, f1_scores, color='blue')
    plt.title('F1-scores vs min_samples_leaf')
    plt.xlabel('Min_samples_leaf')
    plt.ylabel('F1-scores')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    args = create_arg_parser()

    # Load training and testing datasets
    X_train, Y_train = load_data('train_data_it.csv')
    X_test, Y_test = load_data('test_data_it.csv')

    # load vectorizer (tfidf or bag of words)
    vec=get_vector(args)

    # Perform grid search if requested
    if args.grid_search:
        classifier = perform_grid_search(X_train, Y_train, args)
    else:
        # Combine the vectorizer with the classifier indicated in args
        classifier = get_classifier(vec, args)


    # Perform cross validation
    if args.cross_validate:
        stratified_cv(classifier, X_train, Y_train, args.nr_splits, args, random_state=42)

    # Train classifier on training set
    classifier.fit(X_train, Y_train)

    # Print the top features for each class
    if args.top_features:
        print_top_features(classifier, vec, top_n=10)

    # Predict labels for test set
    Y_pred = classifier.predict(X_test)

    # Compute measures
    measures(Y_test, Y_pred)
