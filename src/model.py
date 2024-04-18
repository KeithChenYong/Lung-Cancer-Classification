from crossval import crossval
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def train_model(X, y, config):
    """Train the machine learning model."""
    models = {}

    # Iterate over sections in the config file
    for section in config.sections():
        # Check if the section corresponds to a supported algorithm
        if section == 'LogisticRegression':
            params = dict(config[section])
            models[section] = LogisticRegression(penalty=params['penalty'], C=float(params['c']))
        elif section == 'RandomForestClassifier':
            params = dict(config[section])
            models[section] = RandomForestClassifier(n_estimators=int(params['n_estimators']))
        elif section == 'SVC':
            params = dict(config[section])
            models[section] = SVC(kernel=params['kernel'], C=float(params['c']), gamma=float(params['gamma']))
        else:
            raise ValueError(f"Invalid algorithm: {section}")
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)
    
    # Fit each model separately
    for algo_name, model_instance in models.items():
        model_instance.fit(X_train, y_train)

        # Perform kfold cross validation
        crossval(model_instance, X_train, y_train)

    return models, X_test, y_test

