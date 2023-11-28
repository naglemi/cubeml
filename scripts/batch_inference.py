import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import torch
from gmodetector_py import Hypercube, ImageChannel, FalseColor
from cubeml.model_evaluation import false_color_image

def load_cubelearner_state(file_prefix, save_dir="./):
    # This function exists so we can load pytorch models onto CPU when they were made on GPU
    #   while also loading all other attributes in the CubeLearner object.
    #   We are basically recreating the CubeLearner object by putting these two pieces back together
    #   after running CubeLearner.save_state_for_cpu
    # Define file paths using the provided prefix
    model_path = os.path.join(save_dir, f"{file_prefix}_model_state.pt")
    state_path = os.path.join(save_dir, f"{file_prefix}_learner_state.pkl")

    # Load the learner's state
    with open(state_path, 'rb') as f:
        state = pickle.load(f)

    # Create a new CubeLearner instance with minimal initialization
    learner = CubeLearner(training_data=None, model_type=state['model_type'], init_minimal=True)
    
    # Update the new instance with loaded state
    learner.__dict__.update(state)

    # Initialize the model based on the model_type
    if learner.model_type == "TNN":
        # Assuming the model class is defined within CubeLearner
        model_class = getattr(CubeLearner, "TransformerNN")
        # Assuming n_input_features and num_classes can be inferred from loaded state
        n_input_features = state['features'].shape[1] if state['features'] is not None else None
        num_classes = len(state['labels_dict']) if state['labels_dict'] is not None else None
        learner.model = model_class(n_input_features=n_input_features, num_classes=num_classes)
        learner.model.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        raise ValueError(f"Unknown model type: {learner.model_type}")

    return learner

def load_data(file_path):
    # Extract the directory and file prefix
    file_dir = os.path.dirname(file_path)
    file_prefix = os.path.splitext(os.path.basename(file_path))[0]
    file_extension = os.path.splitext(file_path)[1]

    if file_extension == '.pt':
        # Construct the full paths for .pt and .pkl files
        pt_filepath = os.path.join(file_dir, f"{file_prefix}.pt")
        pkl_filepath = os.path.join(file_dir, f"{file_prefix}.pkl")
        return load_cubelearner_state(file_prefix, file_dir)
    elif file_extension == '.pkl':
        # Loading pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, CubeSchool):
            # Handle CubeSchool object
            return data
        elif isinstance(data, CubeLearner):
            # Handle CubeLearner object
            return data
        else:
            raise ValueError("Unsupported pickle object type.")
    else:
        raise ValueError("Unsupported file type.")

def batch_inference(directory, data_file, method, **kwargs):
    data = load_data(data_file)
    if isinstance(data, CubeSchool):
        # Extract CubeLearner from CubeSchool
        learner = data.learner_dict.get(method)
        if learner is None:
            print(f"Method {method} not found in the provided CubeSchool.")
            return
    elif isinstance(data, CubeLearner):
        learner = data
    else:
        learner = data

    files = os.listdir(directory)
    files = [os.path.join(directory, file) for file in files if string_to_exclude is None or string_to_exclude not in file]

    for filename in files:
        if filename.endswith("_Broadband.hdr"):
            print("Loading img " + filename)
            hypercube_data = Hypercube(filename, min_desired_wavelength=min_wavelength, max_desired_wavelength=max_wavelength)

            inference_map = learner.infer(hypercube_data.hypercube)

            output_filename = filename.replace("_Broadband.hdr", "_segment_uncropped_processed.png")
            output_filename = os.path.join(directory, output_filename)

            # Generate and save the false color image from the segmentation map
            seg_false_color = false_color_image(predictions=inference_map, colors=learner.colors)
            seg_false_color.save(output_filename)

            if false_color:
                # Generate and save the false color image from the original hypercube
                print("Saving out false color RGB in addition to segmentation mask")
                rgb_false_color = FalseColor([
                    ImageChannel(hypercube=hypercube_data, desired_component_or_wavelength=green_wavelength, color='green', cap=green_cap),
                    ImageChannel(hypercube=hypercube_data, desired_component_or_wavelength=red_wavelength, color='red', cap=red_cap),
                    ImageChannel(hypercube=hypercube_data, desired_component_or_wavelength=blue_wavelength, color='blue', cap=blue_cap)
                ])
                
                rgb_false_color.image = rgb_false_color.image.rotate(-90, expand=True)
                
                rgb_false_color.save(os.path.basename(output_filename.replace("_segment_uncropped_processed.png",
                                                                              "_rgb_processed.png")),
                                     output_dir = directory)

            print(f"Inference completed for {filename}. Results saved as {output_filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch inference on a folder of hyperspectral images.")
    parser.add_argument('--dir', type=str, required=True,
                        help='Directory containing hyperspectral images.')
    parser.add_argument('--pickle', type=str, required=True,
                        help='Path to the pickled CubeSchool object.')
    parser.add_argument('--method', type=str, required=True,
                        help='Inference method (e.g., "RF", "PCA").')
    parser.add_argument('--string_to_exclude', type=str,
                        help='String to identify files to be excluded from processing.')
    parser.add_argument('--false_color', action='store_true',
                        help='Generate false color images.')
    parser.add_argument('--green_cap', type = float, default=563,
                        help = "Cap for green channel")
    parser.add_argument('--red_cap', type = float, default=904,
                        help = "Cap for red channel")
    parser.add_argument('--blue_cap', type = float, default=406,
                        help = "Cap for blue channel")
    parser.add_argument('--min_wavelength', type = float, default=400,
                        help = "Minimum desired wavelength for the hypercube")
    parser.add_argument('--max_wavelength', type = float, default=1000,
                        help = "Maximum desired wavelength for the hypercube")
    parser.add_argument('--green_wavelength', type=str, default="533.7419",
                        help='Wavelength for green channel.')
    parser.add_argument('--red_wavelength', type=str, default="563.8288",
                        help='Wavelength for red channel.')
    parser.add_argument('--blue_wavelength', type=str, default="500.0404",
                        help='Wavelength for blue channel.')
    args = parser.parse_args()

    batch_inference(args.dir, args.pickle, args.method, args.string_to_exclude, args.false_color, args.green_cap, args.red_cap, args.blue_cap, args.green_wavelength, args.red_wavelength, args.blue_wavelength)