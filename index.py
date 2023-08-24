import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, json


class RandomForestClassifier:
    def __init__(self, n_estimators=100, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.trees = []

    def fit(self, X, y):
        np.random.seed(self.random_state)

        for _ in range(self.n_estimators):
            sample_indices = np.random.choice(len(X), len(X), replace=True)
            X_subset, y_subset = X[sample_indices], y[sample_indices]
            tree = DecisionTreeClassifier()
            tree.fit(X_subset, y_subset)
            self.trees.append(tree)

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(predictions, axis=0)



class DecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        if self.tree is None:
            raise ValueError("The model has not been trained yet.")
        return [self._predict_single(sample, self.tree) for sample in X]

    def _predict_single(self, sample, tree):
        if isinstance(tree, tuple):
            feature, threshold, left_subtree, right_subtree = tree
            if sample[feature] <= threshold:
                return self._predict_single(sample, left_subtree)
            else:
                return self._predict_single(sample, right_subtree)
        else:
            return tree

    def build_tree(self, X, y):
        if len(set(y)) == 1:
            return y[0]
        if len(X) == 0:
            return np.random.choice(y)
        best_feature, best_threshold = self.find_best_split(X, y)
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        left_tree = self.build_tree(X[left_indices], y[left_indices])
        right_tree = self.build_tree(X[right_indices], y[right_indices])
        return (best_feature, best_threshold, left_tree, right_tree)

    def find_best_split(self, X, y):
        best_gini = float('inf')
        best_feature = None
        best_threshold = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gini = self.compute_gini(X, y, feature, threshold)
                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature
                    best_threshold = threshold
        return best_feature, best_threshold

    def compute_gini(self, X, y, feature, threshold):
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices
        left_gini = self.gini_impurity(y[left_indices])
        right_gini = self.gini_impurity(y[right_indices])
        left_weight = np.sum(left_indices) / len(X)
        right_weight = np.sum(right_indices) / len(X)
        return left_weight * left_gini + right_weight * right_gini

    def gini_impurity(self, y):
        _, counts = np.unique(y, return_counts=True)
        proportions = counts / len(y)
        return 1 - np.sum(proportions ** 2)


class KMeans:
    def __init__(self, n_clusters=8, max_iter=3000, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        np.random.seed(self.random_state)

        # Initialize centroids based on unique quantities
        unique_quantities = np.unique(X[:, 0])  
        initial_indices = []
        for quantity in unique_quantities:
            indices = np.where(X[:, 0] == quantity)[0]
            initial_indices.append(np.random.choice(indices))
        self.centroids = X[initial_indices]

        for _ in range(self.max_iter):
            labels = self.assign_labels(X)

            # Update centroids based on the mean of each feature within the cluster
            new_centroids = np.zeros_like(self.centroids)
            cluster_counts = np.zeros(self.n_clusters)
            for k in range(self.n_clusters):
                cluster_indices = np.where(labels == k)[0]
                cluster_data = X[cluster_indices]
                cluster_counts[k] = len(cluster_data)

                if len(cluster_data) > 0:
                    cluster_mean = np.mean(cluster_data, axis=0)
                    new_centroids[k] = cluster_mean
                else:
                    # If no items in the cluster, keep the current centroid
                    new_centroids[k] = self.centroids[k]

            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break

            self.centroids = new_centroids

        return self

    def assign_labels(self, X):
        distances = np.linalg.norm(X - self.centroids[:, np.newaxis], axis=2)
        return np.argmin(distances, axis=0)

# Data
data = np.array([
  #Number 1 Feature - Quantity
  #Number 2 Feature - Use of Material (Building Material, Tools, Plumbing, Fasteners, Others)
  #Number 3 Feature - Type of Material(0 lagay kapag di building material, Wood, Steel, Concrete, Ceramic)
  #Number 4 Feature - Location(Hidden Pang Train In short san sya)
  #Feature 2 and 3 got swapped

  # Tools
  [3, 2, 0, 1],
  [5, 2, 0, 1],
  [6, 2, 0, 1],
  [10, 2, 0, 1],
  [4, 2, 0, 1],


  # Plumbing
  [1, 3, 0, 2],
  [5, 3, 0, 2],
  [29, 3, 0, 2],
  [3, 3, 0, 2],
  [5, 3, 0, 2],

  #Fasteners
  [5, 4, 0, 3],
  [20 , 4, 0, 3],
  [38 , 4, 0, 3],
  [6 , 4, 0, 3],
  [9 , 4, 0, 3],


  #Others
  [10 , 5, 0, 4],
  [20, 5, 0, 4],
  [51, 5, 0, 4],
  [69, 5, 0, 4],
  [8, 5, 0, 4],

  # Building Material

  # Wood
  [5, 1, 1, 5],
  [20, 1, 1, 5],
  [30, 1, 1, 5],
  [100, 1, 1, 5],
  [10, 1, 1, 5],

  # Steel
  [4, 1, 2, 6],
  [17, 1, 2, 6],
  [33, 1, 2, 6],
  [8, 1, 2, 6],
  [44, 1, 2, 6],

  # Concrete
  [25, 1, 3, 7],
  [30, 1, 3, 7],
  [15, 1, 3, 7],
  [45, 1, 3, 7],
  [50, 1, 3, 7],

  # Ceramic
  [8, 1, 4, 8],
  [78, 1, 4, 8],
  [19, 1, 4, 8],
  [9, 1 , 4, 8],
  [3, 1, 4, 8],

  # Add Quadrant
  # WIP (˃﹏˂✿)
])

features = data[:, :-1]
labels = data[:, -1]


rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(features, labels)

# New data for prediction
new_data = np.array([

    [50, 1, 1],
    [49, 1, 2],
    [40, 5, 0],
    [34, 2, 0],
    [34, 4, 0],
    [105, 5, 0],
    [35, 1, 4],
    [106, 3, 0],
    [36, 5, 0],
    [107, 1, 2],
    [37, 4, 0],
    [108, 2, 0],
    [38, 3, 0],
    [109, 1, 4],
    [39, 3, 0],
    [110, 1, 2],
    [40, 2, 0],
    [111, 1, 1],
    [41, 1, 4],
    [112, 1, 2],
    [42, 2, 0],
    [113, 3, 0],
    [43, 1, 3],
    [114, 1, 4],
    [44, 4, 0],
    [115, 5, 0],
    [45, 4, 0],
    [116, 1, 1],
    [46, 1, 2],
    [117, 1, 4],
    [47, 2, 0],
    [118, 5, 0],
    [48, 1, 1],
    [119, 4, 0],
    [49, 3, 0],
    [120, 1, 3],
    [50, 5, 0],
    [121, 1, 2],
    [51, 3, 0],
    [122, 2, 0],
    [52, 1, 3],
    [123, 2, 0],
    [53, 3, 0],
    [124, 1, 2],
    [54, 5, 0],
    [125, 1, 3],
    [55, 5, 0],
    [126, 1, 4],
    [56, 1, 1],
    [127, 2, 0],
    [57, 1, 3],
    [128, 1, 4],
    [58, 1, 2],
    [129, 1, 1],
    [59, 3, 0],
    [130, 3, 0],
    [60, 1, 2],
    [131, 1, 3],
    [61, 5, 0],
    [132, 3, 0],
    [62, 5, 0],
    [133, 1, 3],
    [63, 4, 0],
    [134, 1, 4],
    [64, 1, 1],
    [135, 2, 0],
    [65, 1, 2],
    [136, 1, 1],
    [66, 2, 0],
    [137, 2, 0],
    [67, 1, 3],
    [138, 1, 4],
    [68, 4, 0],
    [139, 4, 0],
    [69, 1, 4],
    [140, 3, 0],
    [70, 5, 0],
    [141, 2, 0],
    [71, 1, 3],
    [142, 1, 4],
    [72, 3, 0],
    [143, 5, 0],
    [73, 5, 0],
    [144, 1, 4],
    [74, 4, 0],
    [145, 1, 2],
    [75, 3, 0],
    [146, 1, 2],
    [76, 4, 0],
    [147, 1, 3],
    [77, 1, 1],
    [148, 1, 2],
    [78, 2, 0],
    [149, 2, 0],
    [79, 1, 4],
    [150, 5, 0],
    [80, 1, 3],
    [151, 1, 1],
    [81, 2, 0],
    [152, 4, 0],
    [82, 1, 2],
    [153, 5, 0],
    [83, 1, 1],
    [154, 3, 0],
    [84, 1, 4],
    [155, 1, 1],
    [85, 5, 0],
    [156, 3, 0],
    [86, 1, 3],
    [157, 1, 4],
    [87, 1, 2],
    [158, 3, 0],
    [88, 2, 0],
    [159, 4, 0],
    [89, 5, 0],
    [160, 1, 1],
    [90, 3, 0],
    [161, 3, 0],
    [91, 2, 0],
    [162, 4, 0],
    [92, 5, 0],
    [163, 3, 0],
    [93, 1, 4],
    [164, 1, 2],
    [94, 4, 0],
    [165, 5, 0],
    [95, 5, 0],
    [166, 1, 1],
    [96, 4, 0],
    [167, 3, 0],
    [97, 1, 3],
    [168, 5, 0],
    [98, 1, 4],
    [169, 1, 1],
    [99, 5, 0],
    [170, 1, 3],
    [200, 2, 0],
    [171, 4, 0],
    [201, 1, 2],
    [172, 1, 3],
    [202, 2, 0],
    [173, 5, 0],
    [203, 4, 0],
    [174, 3, 0],
    [204, 1, 3],
    [175, 1, 1],
    [205, 5, 0],
    [176, 1, 4],
    [206, 3, 0],
    [177, 1, 4],
    [207, 5, 0],
    [178, 2, 0],
    [208, 3, 0],
    [179, 1, 4],
    [209, 1, 2],
    [180, 5, 0],
    [210, 5, 0],
    [181, 1, 1],
    [211, 4, 0],
    [182, 1, 4],
    [212, 3, 0],
    [183, 5, 0],
    [213, 1, 3],
    [184, 4, 0],
    [214, 2, 0],
    [185, 1, 1],
    [215, 1, 1],
    [186, 2, 0],
    [216, 1, 4],
    [187, 1, 2],
    [217, 1, 3],
    [188, 2, 0],
    [218, 5, 0],
    [189, 4, 0],
    [219, 3, 0],
    [170, 2, 0],
    [220, 1, 2],
    [171, 1, 3],
    [221, 1, 4],
    [172, 4, 0],
    [222, 2, 0],
    [173, 1, 3],
    [223, 3, 0],
    [174, 1, 1],
    [224, 2, 0],
    [175, 5, 0],
    [225, 1, 2],
    [176, 1, 3],
    [226, 4, 0],
    [177, 1, 3],
    [227, 2, 0],
    [178, 1, 1],
    [228, 1, 3],
    [179, 4, 0],
    [229, 1, 2],
    [180, 5, 0],
    [230, 1, 4],
    [181, 2, 0],
    [231, 3, 0],
    [182, 1, 2],
    [232, 5, 0],
    [183, 1, 2],
    [233, 3, 0],
    [184, 4, 0],
    [234, 1, 4],
    [185, 5, 0],
    [235, 1, 3],
    [186, 4, 0],
    [236, 1, 1],
    [187, 3, 0],
    [237, 1, 2],
    [188, 4, 0],
    [238, 1, 1],
    [189, 1, 4],
    [239, 1, 2],
    [190, 5, 0],
    [240, 4, 0],
    [191, 3, 0],
    [241, 2, 0],
    [192, 2, 0],
    [242, 3, 0],
    [193, 1, 1],
    [243, 1, 3],
    [194, 1, 4],
    [244, 4, 0],
    [195, 1, 2],
    [245, 5, 0],
    [196, 1, 4],
    [246, 1, 1],
    [197, 2, 0],
    [247, 3, 0],
    [198, 1, 2],
    [248, 5, 0],
    [199, 1, 4],
    [249, 4, 0],
    [300, 1, 3],
    [250, 3, 0],
    [301, 1, 1],
    [251, 5, 0],
    [302, 2, 0],
    [252, 1, 1],
    [303, 3, 0],
    [253, 1, 3],
    [304, 2, 0],
    [254, 1, 2],
    [305, 5, 0],
    [255, 1, 4],
    [306, 1, 3],
    [256, 3, 0],
    [307, 2, 0],
    [257, 4, 0],
    [308, 1, 4],
    [258, 5, 0],
    [309, 1, 2],
    [259, 1, 4],
    [310, 1, 1],
    [260, 2, 0],
    [311, 1, 3],
    [261, 2, 0],
    [312, 4, 0],
    [262, 1, 4],
    [313, 1, 3],
    [263, 5, 0],
    [314, 2, 0],
    [264, 4, 0],
    [315, 5, 0],
    [265, 1, 1],
    [316, 2, 0],
    [266, 1, 4],
    [317, 1, 3],
    [267, 1, 2],
    [318, 1, 1],
    [268, 1, 3],
    [319, 1, 4],
    [269, 5, 0],
    [320, 2, 0],
    [270, 3, 0],
    [321, 1, 1],
    [271, 1, 4],
    [322, 5, 0],
    [272, 2, 0],
    [323, 1, 4],
    [273, 3, 0],
    [324, 1, 3],
    [274, 1, 2],
    [325, 1, 1],
    [275, 1, 3],
    [326, 2, 0],
    [276, 3, 0],
    [327, 4, 0],
    [277, 1, 2],
    [328, 1, 3],
    [278, 5, 0],
    [329, 2, 0],
    [279, 1, 1],
    [330, 3, 0],
    [280, 2, 0],
    [331, 1, 3],
    [281, 4, 0],
    [332, 2, 0],
    [282, 1, 1],
    [333, 3, 0],
    [283, 1, 3],
    [334, 4, 0],
    [284, 1, 4],
    [335, 5, 0],
    [285, 1, 1],
    [336, 2, 0],
    [286, 3, 0],
    [337, 4, 0],
    [287, 1, 2],
    [338, 1, 3],
    [288, 1, 4],
    [339, 5, 0],
    [289, 1, 2],
    [340, 4, 0],
    [290, 3, 0],
])

# Predict the categories using the trained Random Forest Classifier
predicted_categories = rf_classifier.predict(new_data)
data_with_categories = np.column_stack((new_data, predicted_categories))
print(data_with_categories)

k_clusters = 2

# Using the new_data classified by the Random Forest
new_data_with_categories = np.column_stack((new_data, predicted_categories))
new_data_quantities = new_data_with_categories[:, 0].reshape(-1, 1)

# Initialize centroids using the first few data points
initial_indices = np.random.choice(len(new_data_quantities), size=k_clusters, replace=False)
initial_centroids = new_data_quantities[initial_indices]

# Fit the K-Means model on the original features of the new data
kmeans_model = KMeans(n_clusters=k_clusters, random_state=42)
kmeans_model.fit(new_data)  # Use the original features of new_data
new_data_clusters = kmeans_model.assign_labels(new_data)
centroids = kmeans_model.centroids


app = Flask(__name__)

@app.route("/")
def home_page():
    return render_template('home.html')

@app.route("/get-clusters", methods = ["POST"])
def getClusters():
    return json.dumps({"data": new_data.tolist(), "quadrant": predicted_categories.tolist()})

@app.route("/insert")
def insert_page():
    return render_template('insert.html')

@app.route("/FAQS")
def FAQS_page():
    return render_template('FAQS.html')

@app.route("/clusters", methods = ["GET", "POST"])
def getData():
    
    # test_data = request.get_json(force = True)
    # print(test_data["test_data"])
    # new_data = np.array(test_data["test_data"])
    # kmeans_model = KMeans(n_clusters=k_clusters, random_state=42)
    # kmeans_model.fit(new_data)
    # new_data_clusters = kmeans_model.assign_labels(new_data)

    myclusters = ""
    replenishClusters = ""
    for item, cluster in zip(new_data, new_data_clusters):
        myclusters += "Item: " + str(item) + ", Cluster: " + str(cluster + 1) + "<br />"
    for item, cluster in zip(new_data,new_data_clusters):
        if cluster + 1 == 3:
            replenishClusters += "Item: " + str(item) + ", Cluster: " + str(cluster + 1) + "<br />"


    return render_template("result.html", myclusters=myclusters, replenishClusters=replenishClusters)


if __name__ == '__main__':
   app.run(debug = True)
