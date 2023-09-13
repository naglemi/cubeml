import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from scipy.optimize import minimize
from deap import base, creator, tools, algorithms
import random
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
    "DTC": DTC,
    "TNN": "TNN"
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
        self.optimal_params = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    class TransformerNN(nn.Module):
        def __init__(self, n_input_features, num_classes, d_model=64, num_heads=4, num_dense_layers=3, use_embedding=True):
            super(CubeLearner.TransformerNN, self).__init__()
            self.d_model = d_model
            self.use_embedding = use_embedding
            self.pos_enc = None

            if self.use_embedding:
                self.embedding = nn.Linear(n_input_features, d_model)
                
            self.attention = nn.MultiheadAttention(d_model, num_heads)
            self.norm = nn.LayerNorm(d_model)

            self.denses = nn.ModuleList([nn.Linear(d_model // (2 ** i), d_model // (2 ** (i + 1))) for i in range(num_dense_layers)])
            self.output = nn.Linear(d_model // (2 ** num_dense_layers), num_classes)

        def forward(self, x, pos_enc, verbose="normal"):
            if verbose == "debug":
                print(f"Initial input shape: {x.shape}")

            if verbose == "debug":
                print(f"Initial input shape: {x.shape}")

            if self.use_embedding:
                x = self.embedding(x)
                if verbose == "debug":
                    print(f"After embedding shape: {x.shape}")

            # Get sequence length from the positional encoding size
            seq_len = pos_enc.shape[0]
            if verbose == "debug":
                print(f"Positional encoding sequence length: {seq_len}")

            x = x.unsqueeze(1)
            if verbose == "debug":
                print(f"After unsqueezing: {x.shape}")

            x = x.expand(-1, seq_len, -1)
            if verbose == "debug":
                print(f"After expanding to match positional encoding shape: {x.shape}")

            expanded_pos_enc = pos_enc.unsqueeze(0).expand(x.shape[0], -1, -1)
            if verbose == "debug":
                print(f"Expanded positional encoding shape: {expanded_pos_enc.shape}")

            x = x + expanded_pos_enc
            if verbose == "debug":
                print(f"Shape after adding positional encoding to input: {x.shape}")

            attn_output, _ = self.attention(x, x, x)
            if verbose == "debug":
                print(f"Attention output shape: {attn_output.shape}")

            x = self.norm(x + attn_output)
            if verbose == "debug":
                print(f"Shape after normalization: {x.shape}")

            # Here, we'll aggregate over the sequence length using mean
            # This will reduce the sequence dimension
            x = x.mean(dim=1)
            if verbose == "debug":
                print(f"Shape after mean pooling over sequence length: {x.shape}")

            for dense in self.denses:
                x = nn.ReLU()(dense(x))
                if verbose == "debug":
                    print(f"Shape after dense layer: {x.shape}")

            x = self.output(x)
            if verbose == "debug":
                print(f"Final output shape: {x.shape}")

            return x


    @staticmethod
    def get_positional_encoding(seq_len, d_model):
        positions = torch.arange(seq_len).unsqueeze(1)
        div_terms = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pos_enc = torch.zeros(seq_len, d_model)
        pos_enc[:, 0::2] = torch.sin(positions * div_terms)
        pos_enc[:, 1::2] = torch.cos(positions * div_terms)
        return pos_enc
    
    
    def predict_in_batches(model, data, pos_enc, batch_size, device):
        model.eval()  # Set model to evaluation mode
        outputs = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i: i + batch_size].float().to(device)
            with torch.no_grad():
                output = model(batch_data, pos_enc)
            outputs.append(output)
        return torch.cat(outputs, dim=0)


        
    def fit(self, colors=None, automl = False, verbose = True,
            param_distributions = None,
            param_ranges = None,
            convergence_threshold = 0.001,
            convergence_gen = 5,
            d_model=64, num_heads=4, num_dense_layers=3,
            n_epoch = 500, batch_size = 256,
            **kwargs):
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
        
        if self.model_type == "TNN":
            print("Fitting TNN")
            n_input_features = self.features_train.shape[1]
            
            dataset = TensorDataset(torch.tensor(self.features_train).float(), torch.tensor(self.labels_train).long())
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            test_dataset = TensorDataset(torch.tensor(self.features_test).float(), torch.tensor(self.labels_test).long())
            test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
            
            writer = SummaryWriter()  # initialize the tensorboard writer

            n_input_features = self.features_train.shape[1]
            self.model = CubeLearner.TransformerNN(n_input_features, self.num_classes, d_model=d_model, num_heads=num_heads, num_dense_layers=num_dense_layers)
            self.model.to(self.device) # use GPU if desired

            # Loss and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(self.model.parameters())

            self.model.pos_enc = CubeLearner.get_positional_encoding(n_input_features, d_model).to(self.device)

            for epoch in range(n_epoch):
                # Training phase
                self.model.train()
                total_loss = 0.0
                total_correct = 0
                total_samples = 0
                for batch_features, batch_labels in dataloader:
                    batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)

                    outputs = self.model(batch_features, pos_enc)

                    loss = criterion(outputs, batch_labels)
                    total_loss += loss.item() * batch_features.size(0)  # accumulate batch loss

                    _, predicted = torch.max(outputs, 1)
                    total_correct += (predicted == batch_labels).sum().item()
                    total_samples += batch_labels.size(0)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Calculate average training loss and accuracy over the epoch
                avg_train_loss = total_loss / total_samples
                train_accuracy = total_correct / total_samples

                # Testing phase
                self.model.eval()
                test_total_loss = 0.0
                test_total_correct = 0
                test_total_samples = 0

                with torch.no_grad():
                    for test_batch_features, test_batch_labels in test_dataloader:
                        test_batch_features, test_batch_labels = test_batch_features.to(self.device), test_batch_labels.to(self.device)

                        test_outputs = self.model(test_batch_features, pos_enc)
                        test_loss = criterion(test_outputs, test_batch_labels)
                        test_total_loss += test_loss.item() * test_batch_features.size(0)

                        _, test_predicted = torch.max(test_outputs, 1)
                        test_total_correct += (test_predicted == test_batch_labels).sum().item()
                        test_total_samples += test_batch_labels.size(0)

                avg_test_loss = test_total_loss / test_total_samples
                test_accuracy = test_total_correct / test_total_samples

                # Log to TensorBoard
                writer.add_scalar("Loss/train", avg_train_loss, epoch)
                writer.add_scalar("Accuracy/train", train_accuracy, epoch)
                writer.add_scalar("Loss/test", avg_test_loss, epoch)
                writer.add_scalar("Accuracy/test", test_accuracy, epoch)
                for name, param in self.model.named_parameters():
                    writer.add_histogram(name, param, epoch)

                if verbose:
                    print(f"Epoch {epoch+1}/{n_epoch}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
                    
            # Save predictions for training and test data
            with torch.no_grad():
                # Added lines
                self.results_train = torch.argmax(CubeLearner.predict_in_batches(self.model, torch.tensor(self.features_train), pos_enc, batch_size, self.device), dim=1).cpu().numpy()
                self.results_test = torch.argmax(CubeLearner.predict_in_batches(self.model, torch.tensor(self.features_test), pos_enc, batch_size, self.device), dim=1).cpu().numpy()
                
            writer.close()


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
            
            self.transformed_train = self.model.transform(self.features_train)
            self.transformed_test = self.model.transform(self.features_test)

            if verbose == True:
                print(f"Predicted train labels shape: {self.results_train.shape}")
                print(f"Predicted test labels shape: {self.results_test.shape}")
                print(f"Transformed train data shape: {self.transformed_train.shape}")
                print(f"Transformed test data shape: {self.transformed_test.shape}")


        elif self.model_type in ["RF", "GBC", "ABC", "LR", "GNB", "DTC"]:
            if self.model_type in ["RF", "DTC", "GBC"] and automl in ["grid", "genetic"]:
                print("Debugging here")
                default_param_distributions = {
                    'RF': {
                        'n_estimators': [50,
                                         100,
                                         150],
                        'max_depth': [5,
                                      10, 15,
                                      20],
                        'min_samples_split': [50,
                                              100,
                                              150],
                        'min_samples_leaf': [10,
                                             20,
                                             30],
                        'max_features': ['sqrt', 'log2'],
                        'criterion': ['gini', 'entropy']
                    },
                    'DTC': {
                        'max_depth': [5,
                                      10, 15,
                                      20],
                        'min_samples_split': [50,
                                              100,
                                              150],
                        'min_samples_leaf': [10,
                                             20,
                                             30],
                        'max_features': ['sqrt', 'log2'],
                        'criterion': ['gini', 'entropy']
                    },
                    'GBC': {
                        'n_estimators': [50,
                                         100,
                                         150],
                        'learning_rate': [0.01,
                                          0.05,
                                          0.1],
                        'max_depth': [2,
                                      3,
                                      4],
                        'min_samples_leaf': [10,
                                             20,
                                             30],
                        'max_features': [0.1,
                                         0.3,
                                         0.5]
                    }
                }

                model = models_dict[self.model_type]()


                def objective(params):
                    # Combine current numeric parameters with non-numeric parameters
                    current_params = {**non_numeric_param_ranges, **{k: v for k, v in zip(numeric_param_ranges.keys(), params)}}
                    model.set_params(**current_params)
                    return -np.mean(cross_val_score(model, self.features_train, self.labels_train, cv=3, n_jobs=-1))


                if automl == "grid":
                    if param_distributions is None or not isinstance(param_distributions, dict):
                        param_distributions = default_param_distributions
                        print("Running grid search with default param distributions")
                    grid_search_cv = GridSearchCV(
                        model, param_grid=param_distributions[self.model_type], cv=3, n_jobs=-1
                    )
                    grid_search_cv.fit(self.features_train, self.labels_train)
                    self.optimal_params = grid_search_cv.best_params_
                    self.model = grid_search_cv.best_estimator_  # Assign the model with optimal parameters to self.model
                    print("Best parameters found (in self.optimal_params with grid argument): ", self.optimal_params)
                    
                # For genetic search
                if automl == "genetic":
                    if param_ranges is None:
                        param_ranges = default_param_distributions[self.model_type]
                        print("Running genetic algorithm with default param ranges")

                    # Identify and store fixed parameters
                    fixed_params = {k: v[0] for k, v in param_ranges.items() if len(v) == 1}

                    # Determine type of each parameter and ensure integer ranges are handled correctly
                    int_param_ranges = {k: (int(v[0]), int(v[-1])) for k, v in param_ranges.items() if isinstance(v[0], int) and len(v) > 1}
                    float_param_ranges = {k: v for k, v in param_ranges.items() if isinstance(v[0], float) and len(v) > 1}
                    cat_param_ranges = {k: v for k, v in param_ranges.items() if isinstance(v[0], str) and len(v) > 1}

                    total_params = len(list(int_param_ranges.keys()) + list(float_param_ranges.keys()) + list(cat_param_ranges.keys()))
                    if total_params < 2:
                        raise ValueError("Genetic algorithm requires at least two parameters. Only {} parameter(s) found.".format(total_params))

                    print("Param ranges before separation:", param_ranges)
                    print("Fixed parameters:", fixed_params)
                    print("Total variable parameters:", total_params)
                    print("Integer parameter ranges:", int_param_ranges)
                    print("Float parameter ranges:", float_param_ranges)
                    print("Categorical parameter ranges:", cat_param_ranges)
                    
                    def mutate_integer(individual, indpb, min_value, max_value):
                        for i in range(len(individual)):
                            if random.random() < indpb:
                                individual[i] += random.randint(-abs(max_value - min_value), abs(max_value - min_value))
                                individual[i] = max(min_value, min(individual[i], max_value))

                        return individual,
                    
                    # Sanity check for the length of the individuals
                    def sanity_check_individual(individual):
                        if len(individual) <= 1:
                            raise ValueError("Individuals must contain more than one parameter. Check the parameter ranges and individual creation.")
                            
                    # Define custom crossover function with sanity checks
                    def custom_cxTwoPoint(ind1, ind2):
                        sanity_check_individual(ind1)
                        sanity_check_individual(ind2)
                        return tools.cxTwoPoint(ind1, ind2)
                    
                    def custom_mutation(ind):
                        for i, (key, val) in enumerate(int_param_ranges.items()):
                            min_value, max_value = val
                            toolbox.mutate_integer(ind[i:i+1], min_value=min_value, max_value=max_value)
                        for i, key in enumerate(list(float_param_ranges.keys()), start=len(int_param_ranges)):
                            tools.mutGaussian(ind[i:i+1], mu=0, sigma=1, indpb=0.1)
                        return ind,


                    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
                    creator.create("Individual", list, fitness=creator.FitnessMax)
                    
                    toolbox = base.Toolbox()
                    
                    # Register custom crossover function
                    toolbox.register("mate", custom_cxTwoPoint)

                    for key, val in int_param_ranges.items():
                        toolbox.register(key, random.randint, val[0], val[1])

                    for key, val in float_param_ranges.items():
                        toolbox.register(key, random.uniform, val[0], val[1])

                    for key, val in cat_param_ranges.items():
                        toolbox.register(key, random.choice, val)
                        
                    def init_individual():
                        attributes = []
                        for key in list(int_param_ranges.keys()) + list(float_param_ranges.keys()) + list(cat_param_ranges.keys()):
                            attributes.append(toolbox.__getattribute__(key)())
                        return creator.Individual(attributes)

                    toolbox.register("individual", init_individual)

                    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
                    
                    # Additional sanity checks for individual sizes
                    for ind in toolbox.population(n=50):
                        sanity_check_individual(ind)


                    # Create validation set
                    self.features_train, self.features_val, self.labels_train, self.labels_val = train_test_split(
                        self.features_train, self.labels_train, test_size=0.2, random_state=42)


                    def evalOneMax(individual):
                        params = {k: v for k, v in zip(list(int_param_ranges.keys()) + list(float_param_ranges.keys()) + list(cat_param_ranges.keys()), individual)}
                        model.set_params(**params)
                        model.fit(self.features_train, self.labels_train)
                        # Use validation set for evaluation
                        predictions = model.predict(self.features_val)
                        return f1_score(self.labels_val, predictions, average='micro'),
                    
                    def custom_eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, convergence_threshold = 0.0001, convergence_gen = 5, verbose=__debug__):
                        log = []  # To record stats, if you wish to
                        for gen in range(ngen):
                            
                            print(f"Generation {gen}, Population Size: {len(pop)}")
                            print(f"Fitness values: {[ind.fitness.values[0] for ind in pop if ind.fitness.valid]}")

                            # Step 1: Evaluation of the Population
                            fitnesses = list(map(toolbox.evaluate, pop))
                            for ind, fit in zip(pop, fitnesses):
                                ind.fitness.values = fit

                            # Step 2: Selection
                            selected = toolbox.select(pop, len(pop))

                            # Clone the selected individuals (important to avoid aliasing issues)
                            offspring = list(map(toolbox.clone, selected))

                            # Step 3: Variation
                            # Crossover
                            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                                if random.random() < cxpb:
                                    toolbox.mate(child1, child2)
                                    del child1.fitness.values
                                    del child2.fitness.values

                            # Mutation
                            for mutant in offspring:
                                if random.random() < mutpb:
                                    toolbox.mutate(mutant)
                                    del mutant.fitness.values

                            # Step 4: Re-evaluation
                            fitnesses = list(map(toolbox.evaluate, offspring))
                            for ind, fit in zip(offspring, fitnesses):
                                ind.fitness.values = fit

                            # Step 5: Replacement
                            pop[:] = offspring

                            # Ensure at least one individual has a valid fitness
                            valid_fitnesses = [ind.fitness.values[0] for ind in pop if ind.fitness.valid]
                            if not valid_fitnesses:
                                print(f"Warning: No valid fitness values in generation {gen}.")
                                max_fitness = -np.inf  # or a default value
                            else:
                                max_fitness = max(valid_fitnesses)

                            # Check convergence criteria
                            last_max_scores.pop(0)  # remove oldest score
                            last_max_scores.append(stats.compile(pop)["max"])  # add newest max score

                            if halloffame is not None:  # ensure that halloffame is populated
                                halloffame.update(pop)

                            if np.std(last_max_scores) < convergence_threshold:
                                print("Convergence criteria met. Halting genetic algorithm.")
                                break

                        return pop, log


                    toolbox.register("evaluate", evalOneMax)
                    #toolbox.register("mate", tools.cxTwoPoint) # replaced w custom function
                    toolbox.register("mutate", custom_mutation)
                    # Registering integer mutation function for integer parameters
                    toolbox.register("mutate_integer", mutate_integer, indpb=0.1)

                    toolbox.register("select", tools.selTournament, tournsize=3)

                    pop = toolbox.population(n=50)
                    hof = tools.HallOfFame(1)
                    stats = tools.Statistics(lambda ind: ind.fitness.values)
                    stats.register("avg", np.mean)
                    stats.register("min", np.min)
                    stats.register("max", np.max)
                    
                    last_max_scores = [0] * convergence_gen


                    pop, log = custom_eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.5, ngen=40, stats=stats, halloffame=hof, convergence_threshold = convergence_threshold, verbose=True)

                    optimal_params = {key: val for key, val in zip(list(int_param_ranges.keys()) + list(float_param_ranges.keys()) + list(cat_param_ranges.keys()), hof[0])}
                    
                    # Merge fixed parameters with the optimal parameters found by the genetic algorithm
                    optimal_params_with_fixed = {**fixed_params, **optimal_params}

                    # Set the model's parameters using the merged dictionary
                    model.set_params(**optimal_params_with_fixed)
                    self.model = model
                    self.model.fit(self.features_train, self.labels_train)
                    self.optimal_params = optimal_params_with_fixed
                
                if automl == "grid" or automl == "genetic":
                    print("Best parameters found (in self.optimal_params with either grid or genetic argument): ", self.optimal_params)

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

        elif self.model_type != "TNN":
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

        # Generate positional encoding for TransformerNN
        if hasattr(self, 'model_type') and self.model_type == "TNN":
            seq_len = flattened_data.shape[0]
            d_model = self.model.d_model
            pos_enc = self.model.pos_enc.to(flattened_data.device)

            with torch.no_grad():
                predictions = self.model(flattened_data, pos_enc)
        else:
            predictions = self.model.predict(flattened_data)

        # Reshape the predictions back into the 2D spatial configuration
        inference_map = predictions.reshape(num_rows, num_cols)

        # Return the predictions as a 2D array
        return inference_map


