import os
import glob
import re
import numpy as np
import pandas as pd
from PIL import Image
import json
from gmodetector_py import Hypercube
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import h5py
import warnings

class TrainingData:
    def __init__(self, json_file=None, img_directory=None, png_directory=None, df=None, 
                 normalize_features=False, use_hdf5=False, hdf5_path=None):
        self.json_file = json_file
        self.img_directory = img_directory
        self.png_directory = png_directory
        self.df = df if df is not None else self.load_data()
        self.features = None
        self.labels = None
        self.wavelengths_dict = {} 
        self.labels_key_df = None
        self.labels_char = None
        self.normalize_features = normalize_features
        self.use_hdf5 = use_hdf5
        self.hdf5_path = hdf5_path
        self.create_training_data()
        self.factorize_labels()

    def save_to_hdf5(self):
        """
        Save features and labels to an hdf5 file.
        """
        with h5py.File(self.hdf5_path, 'w') as hf:
            hf.create_dataset('features', data=self.features)
            hf.create_dataset('labels', data=self.labels)

    def load_from_hdf5(self):
        """
        Load features and labels from an hdf5 file.
        """
        with h5py.File(self.hdf5_path, 'r') as hf:
            self.features = np.array(hf['features'])
            self.labels = np.array(hf['labels'])

    def load_data(self):
        """
        Function to load data from json file and return it as a DataFrame.
        """
        with open(self.json_file) as file:
            data = json.load(file)

        df_data = [{'id': entry['id'], 'image': entry['data']['image']} for entry in data]
        df = pd.DataFrame(df_data)
        df['FC_img'] = df['image'].apply(lambda x: x.split('-', 1)[-1])
        df['hdr_img'] = df['FC_img'].str.replace('_falsecolor.png', '.hdr', regex=False)
        df['hdr_img'] = df['hdr_img'].str.replace('.png', '', regex=False)
        df['hdr_img'] = self.img_directory + df['hdr_img']

        return df

    def merge(cls, td_list):
        """
        Merge multiple TrainingData objects into one.
        td_list: list of TrainingData objects
        """
        merged_df = pd.concat([td.df for td in td_list], ignore_index=True)
        return cls(df=merged_df)

    def expand(self, new_df):
        """
        Expand the current TrainingData with a new dataframe.
        new_df: new DataFrame to be added
        """
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        self.create_training_data()
        self.factorize_labels()
        
    def create_training_data(self, use_hdf5=False, hdf5_path="training_data.hdf5", chunk_size=1000):
        """
        Function to create training data from hyperspectral images and PNG files.
        """
        features_list = []
        labels_list = []

        # Set up hdf5 storage
        if use_hdf5:
            hdf5_file = h5py.File(hdf5_path, 'a')
            if 'features' not in hdf5_file:
                maxshape = (None, ) + self.features.shape[1:]
                hdf5_features = hdf5_file.create_dataset("features", shape=(0, ) + self.features.shape[1:], maxshape=maxshape, dtype=self.features.dtype)
                hdf5_labels = hdf5_file.create_dataset("labels", shape=(0,), maxshape=(None,), dtype=self.labels.dtype)
            else:
                hdf5_features = hdf5_file["features"]
                hdf5_labels = hdf5_file["labels"]

        png_files = glob.glob(self.png_directory + "*.png")

        for i in range(len(self.df)):
            row = self.df.loc[i]
            task_id = row['id']

            # Load the hypercube
            test_cube = Hypercube(row['hdr_img'], min_desired_wavelength=400, max_desired_wavelength=1000)

            # Save the wavelengths before converting the hypercube to a numpy array
            self.wavelengths_dict[row['hdr_img']] = test_cube.wavelengths

            hypercube_np = np.array(test_cube.hypercube)
            assert hypercube_np.ndim == 3, f"Hyperspectral image for task {task_id} is not 3-dimensional"

            for png_file in png_files:
                file_task_id, label = extract_task_and_label(png_file)

                if file_task_id == task_id:
                    label_img_np = load_label_image(png_file)

                    # Rotate if necessary
                    if label_img_np.shape != hypercube_np.shape[:2]:
                        label_img_np = np.rot90(np.rot90(np.rot90(label_img_np)))

                    # Select pixels
                    selected_pixels = hypercube_np[label_img_np == 1]

                    # Normalize the pixel values if normalize_features is True
                    if self.normalize_features:
                        selected_pixels = selected_pixels / selected_pixels.max(axis=1, keepdims=True)

                    # Append features and labels
                    features_list.append(selected_pixels)
                    labels_list.extend([label] * selected_pixels.shape[0])

                    # Save to hdf5 if the accumulated data reaches chunk_size
                    if use_hdf5 and len(features_list) >= chunk_size:
                        current_len = hdf5_features.shape[0]
                        hdf5_features.resize(current_len + len(features_list), axis=0)
                        hdf5_features[current_len:] = np.concatenate(features_list, axis=0)
                        hdf5_labels.resize(current_len + len(labels_list), axis=0)
                        hdf5_labels[current_len:] = np.array(labels_list)

                        # Reset the features_list and labels_list
                        features_list = []
                        labels_list = []

        # If there is remaining data in the lists (less than chunk_size), save it to hdf5
        if use_hdf5 and features_list:
            current_len = hdf5_features.shape[0]
            hdf5_features.resize(current_len + len(features_list), axis=0)
            hdf5_features[current_len:] = np.concatenate(features_list, axis=0)
            hdf5_labels.resize(current_len + len(labels_list), axis=0)
            hdf5_labels[current_len:] = np.array(labels_list)

        # If using RAM, concatenate all the data
        elif not use_hdf5:
            self.features = np.concatenate(features_list, axis=0)
            self.labels = np.array(labels_list)

        if use_hdf5:
            # Close hdf5 file after writing all the data
            hdf5_file.close()

    def factorize_labels(self):
        """
        Convert character labels to integer labels and store
        a key to go from integers back to original labels
        """
        # Convert labels to lowercase and sort
        labels_raw = [label.lower() for label in self.labels]
        sorted_labels = np.sort(np.unique(labels_raw))

        # Create a dictionary that maps each label to an integer
        labels_dict = {k: v for v, k in enumerate(sorted_labels)}

        # Map the original labels to integers
        labels_int = [labels_dict[label] for label in labels_raw]

        # Construct label key dataframe
        self.labels_key_df = pd.DataFrame({'Label': sorted_labels, 'Integer': range(len(sorted_labels))})

        # Assign the integer labels
        self.labels = np.array(labels_int)

        # Initialize self.labels_raw with sorted labels
        self.labels_char = sorted_labels



    def plot_spectra(self, colors, sampling_fraction=0.01, hdf5_path=None, chunk_size=1000, verbose: bool = False):
        # If hdf5_path is specified, read from it. Otherwise, assume in-memory.
        if hdf5_path:
            with h5py.File(hdf5_path, 'r') as hdf5_file:
                num_features = hdf5_file["features"].shape[0]
                num_labels = hdf5_file["labels"].shape[0]

                # Sample the features and labels
                num_samples = int(num_features * sampling_fraction)
                sample_indices = np.random.choice(num_features, num_samples, replace=False)

                # Collect sampled features and labels
                sampled_features_list = []
                sampled_labels_list = []
                for idx in sample_indices:
                    feature_chunk_start = (idx // chunk_size) * chunk_size
                    feature_chunk_end = min(feature_chunk_start + chunk_size, num_features)

                    if feature_chunk_start <= idx < feature_chunk_end:
                        offset = idx - feature_chunk_start
                        sampled_features_list.append(hdf5_file["features"][feature_chunk_start:feature_chunk_end][offset])
                        sampled_labels_list.append(hdf5_file["labels"][feature_chunk_start:feature_chunk_end][offset])

                sampled_features = np.array(sampled_features_list)
                sampled_labels = np.array(sampled_labels_list)
        else:
            num_features = len(self.features)
            num_labels = len(self.labels)

            # Sample the features and labels before converting to a dataframe
            num_samples = int(len(self.features) * sampling_fraction)
            sample_indices = np.random.choice(len(self.features), num_samples, replace=False)
            sampled_features = np.array(self.features)[sample_indices]
            sampled_labels = np.array(self.labels)[sample_indices]

        if verbose:
            print(f"Number of features: {num_features}")
            print(f"Number of labels: {num_labels}")

        # Check if all wavelengths are the same
        wavelengths_values = list(self.wavelengths_dict.values())
        if not all(np.array_equal(wavelengths_values[0], wavelengths_value) for wavelengths_value in wavelengths_values):
            raise ValueError("Wavelengths don't match across files")

        # Assuming that all wavelengths match, extract the unique list of wavelengths
        unique_wavelengths = np.array(next(iter(self.wavelengths_dict.values())))

        # Convert unique_wavelengths into a DataFrame
        wavelengths_df = pd.DataFrame(unique_wavelengths, columns=['Wavelength(nm)'])

        # Convert the sampled data to a DataFrame
        df = pd.DataFrame(sampled_features)
        df.columns = wavelengths_df['Wavelength(nm)']
        df['Label'] = sampled_labels

        # dataframe transformation to match the input data format of lineplot function in seaborn
        df_melt = df.melt(id_vars='Label', value_name='Reflectance')

        # Check and convert 'Label' to a categorical variable if not already
        if df_melt['Label'].dtype.name != 'category':
            df_melt['Label'] = df_melt['Label'].astype('category')

        # Check and convert 'Wavelength(nm)' to a float variable if not already
        if df_melt['Wavelength(nm)'].dtype.name != 'float64':
            df_melt['Wavelength(nm)'] = df_melt['Wavelength(nm)'].astype('float64')

        # Define the colormap
        poplar_cmap = ListedColormap(colors, name='organs')

        # Extract the labels list from labels_key_df
        labels_list = self.labels_key_df["Label"].tolist()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=df_melt, x='Wavelength(nm)', y='Reflectance', hue='Label', palette=colors, linewidth=1)
        ax.set_xlim(480, 1000)
        ax.set_ylabel('Intensity', fontsize=8)
        ax.set_xlabel('Wavelength(nm)', fontsize=8)

        # Hide the existing legend
        ax.get_legend().remove()

        # Create a custom legend
        lines = [Line2D([0], [0], color=colors[i], linewidth=3, linestyle='-') for i, label in enumerate(labels_list)]

        ax.legend(lines, labels_list, loc='best', frameon=False, fontsize=7)

        ax.spines['right'].set_visible(False)  # remove right axis
        ax.spines['top'].set_visible(False)  # remove top axis
        plt.tight_layout()
        return fig

    def stratified_sample(self, sample_size=None, hdf5_path=None, chunk_size=1000):
        # If hdf5_path is specified, assume that the data is in an HDF5 file.
        if hdf5_path:
            with h5py.File(hdf5_path, 'r') as hdf5_file:
                features_dataset = hdf5_file['features']
                labels_dataset = hdf5_file['labels']

                # Convert to arrays. Note: This assumes that the whole dataset can fit into memory after sampling.
                # If this isn't the case, more complex chunked processing would be required.
                all_features = np.array(features_dataset)
                all_labels = np.array(labels_dataset)

                # Combine features and labels into a single DataFrame
                data = pd.DataFrame(all_features)
                data['Label'] = all_labels
        else:
            # For in-memory data
            data = pd.DataFrame(self.features)
            data['Label'] = self.labels

        # If sample_size is None, set it to the size of the smallest class
        if sample_size is None:
            sample_size = data['Label'].value_counts().min()
        else:
            # Warn if sample_size is greater than the size of any class
            smallest_class_size = data['Label'].value_counts().min()
            if sample_size > smallest_class_size:
                warnings.warn("Sample size is larger than the size of the smallest class. Some samples will be repeated due to over-sampling.")

        # Perform stratified sampling
        stratified_data = data.groupby('Label').apply(lambda x: x.sample(sample_size, replace=True)).reset_index(drop=True)

        # Update features and labels
        self.features = stratified_data.drop(columns='Label').values
        self.labels = stratified_data['Label'].values

def load_hyperspectral_image(file, min_wavelength=400, max_wavelength=1000):
    """Load a hyperspectral image and return as numpy array."""
    cube = Hypercube(file, min_desired_wavelength=min_wavelength, max_desired_wavelength=max_wavelength)
    return np.array(cube.hypercube)

def load_label_image(file):
    """Load a label image and return as a binary numpy array."""
    img = Image.open(file)
    return np.array(img.convert('1'))

def extract_task_and_label(filename):
    """Extract the task number and label from a filename."""
    task_id = int(re.search(r'task-(\d+)', filename).group(1))
    label = re.search(r'tag-([A-Za-z_]+)', filename).group(1)
    return task_id, label

def match_and_process_images(df, png_directory):
    """Match hyperspectral and label images, process them, and return features and labels."""
    features_list = []
    labels_list = []
    
    png_files = glob.glob(png_directory + "*.png")

    for i in range(len(df)):
        row = df.loc[i]
        hypercube_np = load_hyperspectral_image(row['hdr_img'])

        for png_file in png_files:
            file_task_id, label = extract_task_and_label(png_file)

            if file_task_id == row['id']:
                label_img_np = load_label_image(png_file)

                # Rotate if necessary
                if label_img_np.shape != hypercube_np.shape[:2]:
                    label_img_np = np.rot90(np.rot90(np.rot90(label_img_np)))

                # Select pixels
                selected_pixels = hypercube_np[label_img_np == 1]

                # Append features and labels
                features_list.append(selected_pixels)
                labels_list.extend([label] * selected_pixels.shape[0])

    # Concatenate all the data
    features = np.concatenate(features_list, axis=0)
    labels = np.array(labels_list)
    
    return features, labels

def factorize_labels(labels):
    """Convert labels into factorized integer form."""
    labels_int, labels_key = pd.factorize(labels)
    return labels_int

def generate_training_data(df, png_directory):
    """Generate training data from hyperspectral and label images."""
    training_data = TrainingData()
    training_data.features, training_data.labels = match_and_process_images(df, png_directory)
    training_data.labels = factorize_labels(training_data.labels)
    return training_data
