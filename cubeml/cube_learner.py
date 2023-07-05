from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import time


models_dict = {
    "RF": RF,
    "GBC": GBC,
    "ABC": ABC,
    "LR": LR,
    "GNB": GNB,
    "DTC": DTC
}

class CubeLearner:
    def __init__(self, training_data, model_type, colors=None,
                 train_test=False, test_size=0.2):
        self.training_data = training_data
        self.features = training_data.features
        self.labels = training_data.labels
        self.colors = colors
        self.labels_char = training_data.labels_char
        self.labels_dict = {i: label for i, label in enumerate(self.labels_char)}
        self.model_type = model_type
        self.train_test = train_test
        self.test_size = test_size
        self.model = None
        self.cfx_train = None
        self.cfx_test = None
        self.iou_dict_list_train = []
        self.iou_dict_list_test = []
        self.miou_list_train = []
        self.miou_list_test = []
        self.results = None
        self.results_train = None
        self.results_test = None
        self.labels_train = None
        self.labels_test = None
        self.training_time = None
        self.wavelengths_dict = training_data.wavelengths_dict
        self.feature_names = None
        self.num_classes = np.unique(self.labels).shape[0]  # Assuming labels are numerical and start from 0

    def fit(self, colors=None, automl = False, verbose = False, **kwargs):
        """
        Fit the appropriate model to the training data.
        """

        # Common to all models
        if isinstance(self.features, pd.DataFrame):
            self.feature_names = self.features.columns  # Store column names
            self.features = self.features.to_numpy()  # Convert to numpy array

        self.labels_list = np.unique(self.labels)  # get unique labels

        # Perform train-test split
        self.features_train, self.features_test, self.labels_train, self.labels_test = train_test_split(
            self.features, self.labels, test_size=self.test_size, random_state=42, stratify=self.labels)

        assert len(self.features_train.shape) == len(self.features_test.shape), "Train and test features dimensions do not match"

        start_time = time.time()

        if self.model_type == "PCA":
            self.model = PCA(0.95, **kwargs)  # keep enough components to explain 95% of variance
            self.transformed_train = self.model.fit_transform(self.features)

        elif self.model_type == "LDA":
            n_components = max(2, min(np.unique(self.labels).shape[0] - 1, self.features.shape[1])) # at least 2 or classes - 1 or feature number
            self.model = LDA(n_components=n_components, **kwargs)

            # fit the model and generate predictions
            self.model.fit(self.features_train, self.labels_train)

            self.results_train = self.model.predict(self.features_train)
            self.results_test = self.model.predict(self.features_test)

            # Added lines
            self.transformed_train = self.model.transform(self.features_train)
            self.transformed_test = self.model.transform(self.features_test)

            if verbose == True:
                print(f"Predicted train labels shape: {self.results_train.shape}")
                print(f"Predicted test labels shape: {self.results_test.shape}")
                print(f"Transformed train data shape: {self.transformed_train.shape}")
                print(f"Transformed test data shape: {self.transformed_test.shape}")


        elif self.model_type in ["RF", "GBC", "ABC", "LR", "GNB", "DTC"]:
            if self.model_type in ["RF", "DTC"] and automl in ["grid", "gradient"]:
                param_distributions = {
                    'max_depth': [None, 10, 20, 30, 40, 50],
                    'min_samples_split': [100, 200, 500],
                    'min_samples_leaf': [50, 100, 200],
                }
                param_ranges = {
                    'max_depth': (30, 50),
                    'min_samples_split': (50, 200),
                    'min_samples_leaf': (25, 100),
                }

                if self.model_type == "RF":
                    param_distributions['n_estimators'] = [100, 200, 300, 400, 500]
                    param_ranges['n_estimators'] = (300, 500)
                    model = RF()
                elif self.model_type == "DTC":
                    model = DTC()

                def objective(params):
                    params_int = {key: int(val) for key, val in zip(param_ranges.keys(), params)}
                    model.set_params(**params_int)
                    return -np.mean(cross_val_score(model, self.features_train, self.labels_train, cv=3, n_jobs=-1))

                if automl == "grid":
                    self.model = GridSearchCV(
                        model, param_grid=param_distributions, cv=3, n_jobs=-1
                    )
                    self.model.fit(self.features_train, self.labels_train)
                    print("Best parameters found: ", self.model.best_params_)
                elif automl == "gradient":
                    initial_params = [np.mean(r) for r in param_ranges.values()]
                    result = minimize(objective, initial_params, bounds=param_ranges.values())
                    model.set_params(**{key: int(val) for key, val in zip(param_ranges.keys(), result.x)})
                    self.model = model
                    self.model.fit(self.features_train, self.labels_train)
                    print("Best parameters found: ", {key: int(val) for key, val in zip(param_ranges.keys(), result.x)})

            else:
                self.model = models_dict[self.model_type](**kwargs)
                self.model.fit(self.features_train, self.labels_train)


            # Calculate results_train and results_test for other models
            self.results_train = self.model.predict(self.features_train)
            self.results_test = self.model.predict(self.features_test)

            # For multi-label classification (e.g., a neural network with softmax output), convert class probabilities to class labels
            if len(self.results_train.shape) > 1 and self.results_train.shape[1] > 1:
                self.results_train = np.argmax(self.results_train, axis=1)
                self.results_test = np.argmax(self.results_test, axis=1)

            if verbose == True:
                print("Shape of predictions:", self.results_train.shape)
                print("Type of predictions:", type(self.results_train))
                print("Unique values in predictions:", np.unique(self.results_train))



            # Check for overfitting
            if self.model_type in ["RF", "DTC"]:
                train_score = self.model.score(self.features_train, self.labels_train)
                if train_score > 0.99:
                    print(f'Warning: The {self.model_type} model might overfit. Training score is {train_score}.')
                    print('Consider stweaking model parameters to reduce overfitting.')

        else:
            raise ValueError(f"Invalid model type: {self.model_type}")

        # Generate confusion matrix for the training data for certain models
        if self.model_type != "PCA":
            if verbose == True:
                print("Shape of labels_train:", self.labels_train.shape)
                print("Type of labels_train:", type(self.labels_train))
                print("Unique values in labels_train:", np.unique(self.labels_train))

                print("Shape of results_train:", self.results_train.shape)
                print("Type of results_train:", type(self.results_train))
                print("Unique values in results_train:", np.unique(self.results_train))

            # Generate confusion matrices for all classes
            self.cfx_train = confusion_matrix(self.labels_train, self.results_train, labels=self.labels_list)
            self.cfx_test = confusion_matrix(self.labels_test, self.results_test, labels=self.labels_list)

            # Compute IoU for all classes
            iou_dict_train = self.compute_iou(self.cfx_train)
            iou_dict_test = self.compute_iou(self.cfx_test)

            # Map the class labels to their IoU values
            self.iou_dict_train = {class_label: iou_dict_train[i] for i, class_label in enumerate(self.labels_list)}
            self.iou_dict_test = {class_label: iou_dict_test[i] for i, class_label in enumerate(self.labels_list)}

            # Generate cfx_dict_train
            self.cfx_dict_train = {class_label: self.cfx_train[i] for i, class_label in enumerate(self.labels_list)}

            print(f"Train data confusion matrices for each class: {self.cfx_dict_train}")

            # Predict on test data and compute confusion matrix for test data
            if self.train_test:
                # Generate cfx_dict_test
                self.cfx_dict_test = {class_label: self.cfx_test[i] for i, class_label in enumerate(self.labels_list)}

                print(f"Test data confusion matrices for each class: {self.cfx_dict_test}")


        # Stop timer and calculate elapsed time
        end_time = time.time()
        self.training_time = end_time - start_time  # This is your fitting time in seconds

    
    def compute_iou(self, cfx):
        """Compute Intersection over Union (IoU) for each class given a confusion matrix.

        Args:
            cfx (numpy.ndarray): A confusion matrix.
            num_classes (int): The number of classes.

        Returns:
            dict: A dictionary where the keys are the class indices and the values are the IoU scores.
        """
        iou_dict = {}
        for i in range(self.num_classes):
            tp = cfx[i, i]
            fp = np.sum(cfx[:, i]) - tp
            fn = np.sum(cfx[i, :]) - tp
            iou = tp / (tp + fp + fn)
            iou_dict[i] = iou
        return iou_dict


    def kfold_validate(self, n_splits=5, **kwargs):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=123)

        # Initialize lists to store values
        acc_list_train = []
        acc_list_test = []
        cfx_list_train = []
        cfx_list_test = []

        for train_index, test_index in kf.split(self.features):
            X_train, X_test = self.features[train_index], self.features[test_index]
            y_train, y_test = self.labels[train_index], self.labels[test_index]

            # Create a new instance of the model for each fold
            clf = models_dict[self.model_type](**kwargs)
            clf.fit(X_train, y_train)

            # Predict on train data and compute metrics
            y_predict_train = clf.predict(X_train)
            acc_train = accuracy_score(y_train, y_predict_train)
            acc_list_train.append(acc_train)
            cfx_train = confusion_matrix(y_train, y_predict_train, labels=[0,1,2,3,4], normalize='true')
            cfx_list_train.append(cfx_train)
            iou_dict_train = self.compute_iou(cfx_train)
            self.iou_dict_list_train.append(iou_dict_train)
            miou_train = np.mean(list(iou_dict_train.values()))
            self.miou_list_train.append(miou_train)

            # Predict on test data and compute metrics
            y_predict_test = clf.predict(X_test)
            acc_test = accuracy_score(y_test, y_predict_test)
            acc_list_test.append(acc_test)
            cfx_test = confusion_matrix(y_test, y_predict_test, labels=[0,1,2,3,4], normalize='true')
            cfx_list_test.append(cfx_test)
            iou_dict_test = self.compute_iou(cfx_test)
            self.iou_dict_list_test.append(iou_dict_test)
            miou_test = np.mean(list(iou_dict_test.values()))
            self.miou_list_test.append(miou_test)

            print('Train accuracy: %.3f, Test accuracy: %.3f' % (acc_train, acc_test))
            self.cfx_train = np.mean(cfx_list_train, axis=0)  # Take the mean of the confusion matrices across folds
            self.cfx_test = np.mean(cfx_list_test, axis=0)  # Take the mean of the confusion matrices across folds
            self.plot_cfx(self.cfx_train, self.cfx_test)

        print('Average train accuracy: %.3f, Average test accuracy: %.3f' % (np.mean(acc_list_train), np.mean(acc_list_test)))  # Print the average accuracy over all folds


    def plot_cfx(self, data_type, normalize = True):
        """
        Plot the confusion matrix.
        """
        if data_type == "Train":
            cfx = self.cfx_train
        elif data_type == "Test":
            cfx = self.cfx_test
        else:
            raise ValueError(f"Invalid data_type: {data_type}")

        if cfx is None:
            print("No confusion matrix found. Please fit the model first.")
            return

        if normalize == True:
            # Row normalization
            cfx = cfx.astype('float') / cfx.sum(axis=1)[:, np.newaxis]
            cfx = np.nan_to_num(cfx, copy=True)
            cfx = cfx.round(decimals=2)

        # Create a list of distinct class labels using the labels_dict mapping
        labels_char = [self.labels_dict[i] for i in range(len(self.labels_dict))]

        plt.figure(figsize=(4,3))
        sns.heatmap(cfx, annot=True, cmap='Blues', fmt='g', xticklabels=labels_char, yticklabels=labels_char)
        plt.xlabel('Predicted')
        plt.ylabel('Truth')
        plt.title(f'Confusion Matrix ({self.model_type} - {data_type})') # Added title
        plt.show()


    def gini_importance(self):
        if self.model_type != "RF":
            print("Gini importance is only available for Random Forest models.")
            return
        
        gini = self.model.feature_importances_ 
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(self.training_data.wavelengths_dict['Wavelength(nm)'], gini)
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Gini importance', fontsize=12)
        plt.tight_layout()
        plt.show()


    def visualize(self, data_type = "Train"):
        if data_type not in ["Train", "Test"]:
            raise ValueError("Invalid data_type. Expected 'Train' or 'Test'")

        if self.model_type == "PCA":
            print("PCA\n")
            results = self.transformed_train
            labels = self.labels
            print(f'Results shape: {results.shape}\nLabels shape: {labels.shape}\n')
        elif self.model_type == "LDA":
            if data_type == "Train":
                print("Train\n")
                results = self.transformed_train
                labels = self.labels_train
            else:  # data_type == "Test"
                print("Test\n")
                results = self.transformed_test
                labels = self.labels_test
            print(f'Results shape: {results.shape}\nLabels shape: {labels.shape}\n')
        else:
            if data_type == "Train":
                print("Train\n")
                results = self.results_train
                labels = self.labels_train
            else:  # data_type == "Test"
                print("Test\n")
                results = self.results_test
                labels = self.labels_test
            print(f'Results shape: {results.shape}\nLabels shape: {labels.shape}\n')


        colors = self.colors

        if self.model_type in ["PCA", "LDA"]:
            fig, ax = plt.subplots(figsize=(5, 4))

            s1, s2 = results[:, 0], results[:, 1]

            x_label = 'PC1' if self.model_type == 'PCA' else 'LD1'
            y_label = 'PC2' if self.model_type == 'PCA' else 'LD2'

            if colors:
                cmap = ListedColormap(colors, name='organs')
            else:
                cmap = 'viridis'

            scatter = ax.scatter(s1, s2, c=labels, cmap=cmap, s=20, alpha=0.7)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
            plt.title(f'{self.model_type} Visualization ({data_type})', fontsize=14)
            plt.show()


        elif self.model_type in ["RF", "GBC", "ABC", "LR", "GNB", "DTC"]:
            self.plot_cfx(data_type)

            if self.model_type in ["RF", "GBC", "ABC", "LR"]:
                wavelengths_values = list(self.training_data.wavelengths_dict.values())
                if not all(np.array_equal(wavelengths_values[0], wavelengths_value) for wavelengths_value in wavelengths_values):
                    raise ValueError("Wavelengths don't match across files")

                unique_wavelengths = np.array(next(iter(self.training_data.wavelengths_dict.values())))

                wavelengths_df = pd.DataFrame(unique_wavelengths, columns=['Wavelength(nm)'])

                fig, ax = plt.subplots(figsize=(10, 8))
                ylabel = ''

                if self.model_type == "LR":
                    feature_importance = pd.Series(self.model.coef_[0], index=wavelengths_df['Wavelength(nm)'])
                    ylabel = 'Effect Size'
                elif self.model_type == "RF":
                    if isinstance(self.model, GridSearchCV):
                        feature_importance = pd.Series(self.model.best_estimator_.feature_importances_, index=pd.to_numeric(wavelengths_df['Wavelength(nm)']))
                    else:
                        feature_importance = pd.Series(self.model.feature_importances_, index=pd.to_numeric(wavelengths_df['Wavelength(nm)']))
                    ylabel = 'Gini Importance'
                elif self.model_type == "GBC":
                    if isinstance(self.model, GridSearchCV):
                        feature_importance = pd.Series(self.model.best_estimator_.feature_importances_, index=pd.to_numeric(wavelengths_df['Wavelength(nm)']))
                    else:
                        feature_importance = pd.Series(self.model.feature_importances_, index=pd.to_numeric(wavelengths_df['Wavelength(nm)']))
                    ylabel = 'Gradient Boosting Importance'
                elif self.model_type == "ABC":
                    if isinstance(self.model, GridSearchCV):
                        feature_importance = pd.Series(self.model.best_estimator_.feature_importances_, index=pd.to_numeric(wavelengths_df['Wavelength(nm)']))
                    else:
                        feature_importance = pd.Series(self.model.feature_importances_, index=pd.to_numeric(wavelengths_df['Wavelength(nm)']))
                    ylabel = 'AdaBoost Importance'
                else:
                    if isinstance(self.model, GridSearchCV):
                        feature_importance = pd.Series(self.model.best_estimator_.feature_importances_, index=pd.to_numeric(wavelengths_df['Wavelength(nm)']))
                    else:
                        feature_importance = pd.Series(self.model.feature_importances_, index=pd.to_numeric(wavelengths_df['Wavelength(nm)']))
                    ylabel = 'Importance'

                feature_importance.plot(ax=ax, color='b')
                ax.set_xlabel('Wavelength (nm)', fontsize=12)
                ax.set_ylabel(ylabel, fontsize=12)
                plt.title(f'{self.model_type} {ylabel} Over Wavelength', fontsize=14)
                plt.show()


    def infer(self, hypercube_data):
        # Flatten the 3D hypercube into 2D so we can run our classifier on it
        num_rows, num_cols, num_bands = hypercube_data.shape
        flattened_data = hypercube_data.reshape(num_rows * num_cols, num_bands)

        # Use our trained model to make predictions
        predictions = self.model.predict(flattened_data)

        # Reshape the predictions back into the 2D spatial configuration
        inference_map = predictions.reshape(num_rows, num_cols)

        # Return the predictions as a 2D array
        return inference_map
