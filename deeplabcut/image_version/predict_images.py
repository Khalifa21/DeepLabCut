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

####################################################
# Dependencies
####################################################

import os.path
from deeplabcut.pose_estimation_tensorflow.nnet import predict
from deeplabcut.pose_estimation_tensorflow.config import load_config
import time
import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import tensorflow as tf
from deeplabcut.utils import auxiliaryfunctions
from skimage.util import img_as_ubyte
import scipy
import matplotlib.pyplot as plt


####################################################
# Loading data, and defining model folder
####################################################

def GetPosesofFrames(cfg, dlc_cfg, sess, inputs, outputs, directory, output_directory, framelist, nframes, batchsize,
                     rgb, max_input_size):
    ''' Batchwise prediction of pose  for framelist in directory'''
    from skimage import io
    print("Starting to extract posture")
    if rgb:
        im = io.imread(os.path.join(directory, framelist[0]), mode='RGB')
    else:
        im = io.imread(os.path.join(directory, framelist[0]))

    ny, nx, nc = np.shape(im)
    print("Overall # of frames: ", nframes, " found with (before cropping) frame dimensions: ", nx, ny)

    PredicteData = np.zeros((nframes, 3 * len(dlc_cfg['all_joints_names'])))
    batch_ind = 0  # keeps track of which image within a batch should be written to
    batch_num = 0  # keeps track of which batch you are at

    if cfg['cropping']:
        print(
            "Cropping based on the x1 = %s x2 = %s y1 = %s y2 = %s. You can adjust the cropping coordinates in the config.yaml file." % (
                cfg['x1'], cfg['x2'], cfg['y1'], cfg['y2']))
        nx, ny = cfg['x2'] - cfg['x1'], cfg['y2'] - cfg['y1']
        if nx > 0 and ny > 0:
            pass
        else:
            raise Exception('Please check the order of cropping parameter!')
        if cfg['x1'] >= 0 and cfg['x2'] < int(np.shape(im)[1]) and cfg['y1'] >= 0 and cfg['y2'] < int(np.shape(im)[0]):
            pass  # good cropping box
        else:
            raise Exception('Please check the boundary of cropping!')

    pbar = tqdm(total=nframes)
    counter = 0
    step = max(10, int(nframes / 100))

    if max_input_size == None:
        pose_config_file = Path(
            str(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pose_cfg.yaml'))).resolve()
        pose_config = auxiliaryfunctions.read_plainconfig(pose_config_file)
        print("Pose_cfg file read successfully.")
        max_input_size = pose_config['max_input_size'] - 1

    if batchsize == 1:
        for counter, framename in enumerate(framelist):
            # frame=io.imread(os.path.join(directory,framename),mode='RGB')
            if rgb:
                im = io.imread(os.path.join(directory, framename), mode='RGB')
            else:
                im = io.imread(os.path.join(directory, framename))

            (x, y, _) = im.shape
            max_size = max(x, y)
            scale_factor = max_input_size / max_size
            scaled_image = scipy.misc.imresize(im, scale_factor)
            framename = os.path.join(output_directory, framename)
            scipy.misc.imsave(framename, scaled_image)
            im = scaled_image

            if counter % step == 0:
                pbar.update(step)

            if cfg['cropping']:
                frame = img_as_ubyte(im[cfg['y1']:cfg['y2'], cfg['x1']:cfg['x2'], :])
            else:
                frame = img_as_ubyte(im)

            pose = predict.getpose(frame, dlc_cfg, sess, inputs, outputs)
            PredicteData[counter, :] = pose.flatten()
    else:
        frames = np.empty((batchsize, ny, nx, 3), dtype='ubyte')  # this keeps all the frames of a batch
        for counter, framename in enumerate(framelist):
            if rgb:
                im = io.imread(os.path.join(directory, framename), mode='RGB')
            else:
                im = io.imread(os.path.join(directory, framename))

            (x, y, _) = im.shape
            max_size = max(x, y)
            scale_factor = max_input_size / max_size
            scaled_image = scipy.misc.imresize(im, scale_factor)
            framename = os.path.join(output_directory, framename)
            scipy.misc.imsave(framename, scaled_image)
            im = scaled_image

            if counter % step == 0:
                pbar.update(step)

            if cfg['cropping']:
                frames[batch_ind] = img_as_ubyte(im[cfg['y1']:cfg['y2'], cfg['x1']:cfg['x2'], :])
            else:
                frames[batch_ind] = img_as_ubyte(im)

            if batch_ind == batchsize - 1:
                pose = predict.getposeNP(frames, dlc_cfg, sess, inputs, outputs)
                PredicteData[batch_num * batchsize:(batch_num + 1) * batchsize, :] = pose
                batch_ind = 0
                batch_num += 1
            else:
                batch_ind += 1

        if batch_ind > 0:  # take care of the last frames (the batch that might have been processed)
            pose = predict.getposeNP(frames, dlc_cfg, sess, inputs,
                                     outputs)  # process the whole batch (some frames might be from previous batch!)
            PredicteData[batch_num * batchsize:batch_num * batchsize + batch_ind, :] = pose[:batch_ind, :]

    pbar.close()
    return PredicteData, nframes, nx, ny


def analyze_images(config, directory, image_type='.png', shuffle=1, trainingsetindex=0, gputouse=None,
                   save_as_csv=False, rgb=True, max_input_size=None):
    """
    Analyzed all images (of type = image_type) in a folder and stores the output in one file.

    You should set max_input_size so all images larger than that will be resized to match that number,
    if not set, default value from pose_config file will be used.

    You can crop the frames (before analysis), by changing 'cropping'=True and setting 'x1','x2','y1','y2' in the config file.

    Output: The labels are stored as MultiIndex Pandas Array, which contains the name of the network, body part name, (x, y) label position \n
            in pixels, and the likelihood for each frame per body part. These arrays are stored in an efficient Hierarchical Data Format (HDF) \n
            in prediction directory, where the resized images are stored. However, if the flag save_as_csv is set to True, the data can also be exported in \n
            comma-separated values format (.csv), which in turn can be imported in many programs, such as MATLAB, R, Prism, etc.

    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    directory: string
        Full path to directory containing the frames that shall be analyzed

    frametype: string, optional
        Checks for the file extension of the frames. Only images with this extension are analyzed. The default is ``.png``

    shuffle: int, optional
        An integer specifying the shuffle index of the training dataset used for training the network. The default is 1.

    trainingsetindex: int, optional
        Integer specifying which TrainingsetFraction to use. By default the first (note that TrainingFraction is a list in config.yaml).

    gputouse: int, optional. Natural number indicating the number of your GPU (see number in nvidia-smi). If you do not have a GPU put None.
    See: https://nvidia.custhelp.com/app/answers/detail/a_id/3751/~/useful-nvidia-smi-queries

    save_as_csv: bool, optional
        Saves the predictions in a .csv file. The default is ``False``; if provided it must be either ``True`` or ``False``

    rbg: bool, optional.
        Whether to load image as rgb; Note e.g. some tiffs do not alow that option in io.imread, then just set this to false.

    max_input_size : Integer
        Integer represents the maximum size for the images that will be used for prediction,
        so any image that exceeds this number will be resized. In case it is not set or set with none,
        the value from pose_config file will be used.

    Examples
    --------
    A full example is provided at test_example file.
    """
    if 'TF_CUDNN_USE_AUTOTUNE' in os.environ:
        del os.environ['TF_CUDNN_USE_AUTOTUNE']  # was potentially set during training

    tf.reset_default_graph()
    start_path = os.getcwd()  # record cwd to return to this directory in the end

    cfg = auxiliaryfunctions.read_config(config)
    trainFraction = cfg['TrainingFraction'][trainingsetindex]
    modelfolder = os.path.join(cfg["project_path"], str(auxiliaryfunctions.GetModelFolder(trainFraction, shuffle, cfg)))
    path_test_config = Path(modelfolder) / 'test' / 'pose_cfg.yaml'
    try:
        dlc_cfg = load_config(str(path_test_config))
    except FileNotFoundError:
        raise FileNotFoundError(
            "It seems the model for shuffle %s and trainFraction %s does not exist." % (shuffle, trainFraction))

    # Check which snapshots are available and sort them by # iterations
    try:
        Snapshots = np.array(
            [fn.split('.')[0] for fn in os.listdir(os.path.join(modelfolder, 'train')) if "index" in fn])
    except FileNotFoundError:
        raise FileNotFoundError(
            "Snapshots not found! It seems the dataset for shuffle %s has not been trained/does not exist.\n Please train it before using it to analyze images.\n Use the function 'train_network' to train the network for shuffle %s." % (
            shuffle, shuffle))

    if cfg['snapshotindex'] == 'all':
        print(
            "Snapshotindex is set to 'all' in the config.yaml file. Running images analysis with all snapshots is very costly! Use the function 'evaluate_network' to choose the best the snapshot. For now, changing snapshot index to -1!")
        snapshotindex = -1
    else:
        snapshotindex = cfg['snapshotindex']

    increasing_indices = np.argsort([int(m.split('-')[1]) for m in Snapshots])
    Snapshots = Snapshots[increasing_indices]

    print("Using %s" % Snapshots[snapshotindex], "for model", modelfolder)

    ##################################################
    # Load and setup CNN part detector
    ##################################################

    # Check if data already was generated:
    dlc_cfg['init_weights'] = os.path.join(modelfolder, 'train', Snapshots[snapshotindex])
    trainingsiterations = (dlc_cfg['init_weights'].split(os.sep)[-1]).split('-')[-1]

    # update batchsize (based on parameters in config.yaml)
    dlc_cfg['batch_size'] = 1  # cfg['batch_size']

    # Name for scorer:
    DLCscorer = auxiliaryfunctions.GetScorerName(cfg, shuffle, trainFraction, trainingsiterations=trainingsiterations)
    sess, inputs, outputs = predict.setup_pose_prediction(dlc_cfg)
    pdindex = pd.MultiIndex.from_product([[DLCscorer], dlc_cfg['all_joints_names'], ['x', 'y', 'likelihood']],
                                         names=['scorer', 'bodyparts', 'coords'])

    if gputouse is not None:  # gpu selectinon
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gputouse)

    ##################################################
    # Loading the images
    ##################################################
    # checks if input is a directory
    if os.path.isdir(directory) == True:
        """
        Analyzes all the frames in the directory.
        """
        print("Analyzing all frames in the directory: ", directory)
        os.chdir(directory)
        output_directory = os.path.join(directory, cfg['Task']+'_'+"prediction")
        framelist = np.sort([fn for fn in os.listdir(os.curdir) if (image_type in fn)])

        vname = Path(directory).stem
        dataname = os.path.join(output_directory, vname + DLCscorer + '.h5')
        try:
            # Attempt to load data...
            pd.read_hdf(dataname)
            print("Frames already analyzed!", dataname)
        except FileNotFoundError:
            os.mkdir(output_directory)
            nframes = len(framelist)
            if nframes >= 1:
                start = time.time()
                PredicteData, nframes, nx, ny = GetPosesofFrames(cfg, dlc_cfg, sess, inputs, outputs, directory,
                                                                 output_directory, framelist, nframes,
                                                                 dlc_cfg['batch_size'], rgb, max_input_size)
                stop = time.time()

                if cfg['cropping'] == True:
                    coords = [cfg['x1'], cfg['x2'], cfg['y1'], cfg['y2']]
                else:
                    coords = [0, nx, 0, ny]

                dictionary = {
                    "start": start,
                    "stop": stop,
                    "run_duration": stop - start,
                    "Scorer": DLCscorer,
                    "config file": dlc_cfg,
                    "batch_size": dlc_cfg["batch_size"],
                    "frame_dimensions": (ny, nx),
                    "nframes": nframes,
                    "cropping": cfg['cropping'],
                    "cropping_parameters": coords
                }
                metadata = {'data': dictionary}

                print("Saving results in %s..." % (output_directory))
                os.chdir(output_directory)
                auxiliaryfunctions.SaveData(PredicteData[:nframes, :], metadata, dataname, pdindex, framelist,
                                            save_as_csv)
                print("The folder was analyzed. Now your research can truly start!")
                print("If the tracking is not satisfactory for some frome, consider expanding the training set.")
            else:
                print("No frames were found. Consider changing the path or the frametype.")

    os.chdir(str(start_path))


def get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def plotting(labels_path_file: str, scorer: str = None, Labels = ['o','X','D'], dotsize = 10, alpha=0.7):

    labels_path = Path(labels_path_file)
    if labels_path.exists() and labels_path.is_file():
        dir = labels_path.parents[0]
        labeled_data = get_labels_data(labels_path)
        scorers = []
        if labeled_data is None:
            raise TypeError('File of labels is not of supported type: ' + str(labels_path.suffix))
        if scorer is not None and scorer not in labeled_data.columns.get_level_values('scorer'):
            raise ValueError('Scorer not found in file of labels: ' + scorer)
        else:
            scorers.append(scorer)
        if scorer is None:
            scorers = labeled_data.columns.get_level_values('scorer').unique()
            if len(scorers) < 1:
                raise ValueError('No scorer contained in file of labels, therefor invalid file.')
        bodyparts = labeled_data.columns.get_level_values('bodyparts').unique()
        cmap = get_cmap(len(bodyparts))

        tmpfolder = str(dir) + '_labeled'
        auxiliaryfunctions.attempttomakefolder(tmpfolder)
        for image_index, image_name in enumerate(labeled_data.index):
            for scorer_index, current_scorer in enumerate(scorers):
                if scorer is None or current_scorer == scorer:
                    image = io.imread(dir.joinpath(image_name))
                    #plt.axis('off')

                    if np.ndim(image) == 2:
                        h, w = np.shape(image)
                    else:
                        h, w, nc = np.shape(image)

                    plt.figure(
                        frameon=False, figsize=(w * 1. / 100 , h * 1. / 100))
                    plt.subplots_adjust(
                        left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

                    plt.imshow(image, 'gray')

                    for c, bp in enumerate(bodyparts):
                        plt.plot(
                            labeled_data[current_scorer][bp]['x'].values[image_index],
                            labeled_data[current_scorer][bp]['y'].values[image_index],
                            Labels[scorer_index],
                            color=cmap(c),
                            alpha=alpha,
                            ms=dotsize)

                    plt.xlim(0, w)
                    plt.ylim(0, h)
                    #plt.axis('off')
                    plt.subplots_adjust(
                        left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
                    plt.gca().invert_yaxis()

                    plt.savefig(str(Path(tmpfolder) / image_name.split(os.sep)[-1]))  # create file name
                    plt.close("all")

    else:
        raise FileNotFoundError('File of labels not found or path is not a file: ' + str(labels_path))

def get_labels_data(labels_path: Path):
    '''Returns labeled_data under path

        Input: labels_path of Path from pathlib

        Output: labeled data in Dataframe pandas or None if type not supported
    '''
    labels = None
    if labels_path.suffix == '.h5':
        labels = pd.read_hdf(str(labels_path))
    elif labels_path.suffix == '.csv':
        labels = pd.read_csv(str(labels_path))
    elif labels_path.suffix == '.pickle':
        labels = pd.read_pickle(str(labels_path))
    return labels

