"""
images_version is a contribution to DeepLabCut2.0 Toolbox
https://github.com/AlexEMG/DeepLabCut
DeepLabCut authors:
A Mathis, alexander.mathis@bethgelab.org
T Nath, nath@rowland.harvard.edu
M Mathis, mackenzie@post.harvard.edu
Images_version authors:
Mohamed Ahmed, mohammed.khlifa21@gmail.com
Thomas Dittrich-Salamon, hanniball1998@gmail.com
"""
from deeplabcut.image_version.new import create_new_project
from deeplabcut.image_version.frame_extraction import extract_frames
from deeplabcut.image_version.trainingsetmanipulation import label_frames, check_labels,create_training_dataset
from deeplabcut.image_version import training, evaluate
from deeplabcut.image_version.predict_images import analyze_images

task = 'test_project3'
experimenter = 'Mohamed' ## put your name here
images_dir = ['/Users/mohamed/cv_images_train'] ## images used for training and validation
predict_dir = '/Users/mohamed/cv_images_test' ## images used for prediction

# This step is to create the project directories and the config file that is used for the whole project.
# Don't forget to change config file data for your project.
config_file_path = create_new_project(task, experimenter, images_dir,working_directory='/Users/mohamed/cv_projects', copy_images=True, image_type='.jpg')

# config_file_path will be none in case project already exists.
# change task/experimenter variable or delete project directory if you want to run the test again
if config_file_path is not None:

    # This step is to stay consistent with DeepLabCut directory structure
    # and to resize images that are larger than max_input_size.
    extract_frames(config_file_path, max_input_size=800)

    # This step is to use the labeling tool to label your data.
    # You will find your data under random named directory after clicking "load data".
    label_frames(config_file_path)

    # This step is to check if the labels were created and stored correctly.
    check_labels(config_file_path)

    # This function sums all data sets and make them ready for training.
    create_training_dataset(config_file_path)

    # This function trains the network, maxiters is set to 5 for the sake of example.
    training.train_network(config_file_path, maxiters=5)

    # This function is to evaluate the network, don't forget plotting parameter to show the labels on the images.
    evaluate.evaluate_network(config_file_path, plotting=True)

    # This function is to predict on new images. You will find the result in "prediction" subdirectory.
    analyze_images(config_file_path, predict_dir,image_type='.jpg', save_as_csv=True, max_input_size=800)
