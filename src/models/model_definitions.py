from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier


RANDOM_STATE = 42


def get_models():
    
    models = {
        
        "Logistic Regression": LogisticRegression(
            max_iter=10000,
            solver="saga",
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        
        "Naive Bayes": GaussianNB(),
        
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100,),
            max_iter=1000,
            random_state=RANDOM_STATE
        )
    }
    
    return models