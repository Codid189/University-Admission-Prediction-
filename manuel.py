import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class AdmissionDecisionTree:
    def __init__(self, max_depth=5):
        self.max_depth = max_depth
        self.root = None

    def entropy(self, y):
        # Calculate entropy
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def information_gain(self, parent, left_child, right_child):
        # Calculate information gain
        parent_entropy = self.entropy(parent)
        left_entropy = self.entropy(left_child)
        right_entropy = self.entropy(right_child)
        
        n = len(parent)
        left_weight = len(left_child) / n
        right_weight = len(right_child) / n
        
        return parent_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def split(self, X, y, feature, threshold):
        # Split data based on feature and threshold
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def build_tree(self, X, y, depth=0):
        # Recursive tree building
        n_samples, n_features = X.shape
        
        # Stopping conditions
        if (depth >= self.max_depth or 
            len(np.unique(y)) == 1 or 
            n_samples < 2):
            # Return leaf node with majority class
            return np.round(np.mean(y))

        # Find best split
        best_gain = -1
        best_feature, best_threshold = None, None

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                # Try splitting on this feature and threshold
                X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
                
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                # Calculate information gain
                gain = self.information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        # If no good split found, return majority class
        if best_gain == -1:
            return np.round(np.mean(y))

        # Recursive split
        X_left, X_right, y_left, y_right = self.split(X, y, best_feature, best_threshold)

        # Create tree node
        tree = {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': self.build_tree(X_left, y_left, depth + 1),
            'right': self.build_tree(X_right, y_right, depth + 1)
        }
        return tree

    def fit(self, X, y):
        # Build the decision tree
        self.root = self.build_tree(X, y)
        return self

    def predict_single(self, x):
        # Predict for a single sample
        def traverse_tree(node, x):
            # If it's a leaf node (just a value)
            if not isinstance(node, dict):
                return node
            
            # Get feature and threshold for splitting
            feature = node['feature']
            threshold = node['threshold']
            
            # Decide which branch to follow
            if x[feature] <= threshold:
                return traverse_tree(node['left'], x)
            else:
                return traverse_tree(node['right'], x)
        
        return traverse_tree(self.root, x)

    def predict(self, X):
        # Predict for multiple samples
        return np.array([self.predict_single(x) for x in X])

# Load and prepare data
def load_admission_data(filepath):
    # Load the CSV file
    df = pd.read_csv('datadash.csv')
    
    # Prepare features and target
    X = df[['GRE_Score', 'LOR', 'CGPA']].values
    y = df['Chance_of_Admit'].values
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

# Main execution
filepath = 'admission_data.csv'  # Replace with your actual file path
X, y = load_admission_data(filepath)

# Binary classification: convert to 0 or 1
y_binary = (y >= 0.5).astype(int)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Create and train the decision tree
tree = AdmissionDecisionTree(max_depth=3)
tree.fit(X_train, y_train)

# Predict and evaluate
y_pred = tree.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2%}")