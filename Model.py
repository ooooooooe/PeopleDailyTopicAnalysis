from sklearn.cluster import KMeans

class Kmeans:
    def __init__(self, centers = 2):
        self.model = KMeans(n_clusters = centers)

    def predict(self, data):
        self.model.fit(data)

    def showResults(self):
        print(self.model.labels_)
        print(self.model.inertia_)

    @staticmethod
    def multiPredict(data, leftRange, rightRange, title=None):
        for i in range(leftRange, rightRange):
            print("Training with cluster number %d ..." %i )
            model = KMeans(n_clusters = i)
            model.fit(data)
            print(i, model.inertia_)
