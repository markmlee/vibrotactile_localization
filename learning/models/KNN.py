import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, root_mean_squared_error, mean_absolute_error

class KNN:
    def __init__(self):
        
        neighbors_count = 5

        self.knn = KNeighborsRegressor(n_neighbors=neighbors_count)

    def fit(self, x_train, y_train):
        self.knn.fit(x_train, y_train)

    def predict(self, x):
        return self.knn.predict(x)

    def score(self, x, y):
        return self.knn.score(x, y)

    def rmse(self, x, y):
        y_pred = self.knn.predict(x)

        #convert tensor to numpy array
        y = y.numpy()


        #use only the first element of y_pred, and first element of y
        y_pred = y_pred[:,0]*100
        y = y[:,0]*100

        print(f"y_pred: {y_pred}, y: {y}")
        print(f"y_diff: {y_pred-y}")


        #compute the RMSE of y and y_pred
        y_diff_mse = np.mean((y_pred - y)**2)
        y_diff_rmse = np.sqrt(y_diff_mse)


        print(f"y_diff_rmse: {y_diff_rmse}")

        rmse = root_mean_squared_error(y, y_pred)
        print(f"rmse: {rmse}")

        return rmse
    
    def mae(self, x, y):
        y_pred = self.knn.predict(x)

        #convert tensor to numpy array
        y = y.numpy()


        #use only the first element of y_pred, and first element of y
        y_pred = y_pred[:,0]*100
        y = y[:,0]*100

        print(f"y_pred: {y_pred}, y: {y}")
        print(f"y_diff: {y_pred-y}")


        #compute the MAE of y and y_pred
        y_diff_mae = np.mean(np.abs(y_pred - y))

        print(f"y_diff_mae: {y_diff_mae}")

        mae = mean_absolute_error(y, y_pred)
        print(f"mae: {mae}")

        y_pred_list = []
        y_list = []

        y_pred_list.extend(y_pred)
        y_list.extend(y)

        return mae, y_pred_list, y_list

    def save(self, path):
        np.save(path, self.knn)

    def load(self, path):
        self.knn = np.load(path)
        return self.knn