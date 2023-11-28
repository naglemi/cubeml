import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse
import torch
from gmodetector_py import Hypercube, ImageChannel, FalseColor
from cubeml.model_evaluation import false_color_image

# Custom unpickler function
def custom_unpickler(file_path):
    try:
        return pickle.load(file_path)
    except Exception as e:
        print(f"Standard loading failed due to {e}. Trying with map_location...")
        return torch.load(file_path, map_location=torch.device('cpu'))

def batch_inference(directory, school_pickle, method, string_to_exclude=None, false_color=False, green_cap=563, red_cap=904, blue_cap=406, green_wavelength="533.7419", red_wavelength="563.8288", blue_wavelength="500.0404", min_wavelength=400, max_wavelength=1000):
    with open(school_pickle, 'rb') as f:
        school = custom_unpickler(f)

    if method not in school.learner_dict:
        print(f"Method {method} not found in the provided CubeSchool.")
        return

    learner = school.learner_dict[method]

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
