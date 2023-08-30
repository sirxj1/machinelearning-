import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X, y)
tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print("Decision Tree Rules:\n", tree_rules)
