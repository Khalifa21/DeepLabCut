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


def extract_frames(config, max_input_size=None):
    """
    This step exist to follow the same test and directory structure of original DeepLabCut and
    also to resize the images according to max_input_size.
    Parameters
    ----------
    config : string
        Full path of the config.yaml file as a string.

    max_input_size : Integer
        Integer represents the maximum size for the images that will be extracted for labeling,
        so any image that exceeds this number will be resized. In case it is not set or set with none,
        the value from pose_config file will be used.

    Examples
    --------
    A full example is provided at test_example file.
    """
    import os
    from pathlib import Path
    from deeplabcut.image_version import auxiliaryfunctions
    import scipy
    import pandas as pd

    config_file = Path(config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")

    if max_input_size == None:
        pose_config_file = Path(str(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'pose_cfg.yaml'))).resolve()
        pose_config = auxiliaryfunctions.read_plainconfig(pose_config_file)
        print("Pose_cfg file read successfully.")
        max_input_size = pose_config['max_input_size'] - 1

    images_sets = cfg['images_sets'].keys()

    for image_set_indx,image_set in enumerate(images_sets):
        fname = Path(image_set)
        output_path = Path(config).parents[0] / 'labeled-data' / fname.stem
        output_path.mkdir(parents=True, exist_ok=True)
        images=[os.path.join(image_set,vp) for vp in os.listdir(image_set)]
        destinations = [os.path.join(output_path, vp) for vp in os.listdir(image_set)]
        csv_file = os.path.join(output_path, 'CollectedData_' + cfg['scorer'] + '.csv')
        hd5_file = os.path.join(output_path, 'CollectedData_' + cfg['scorer'] + '.h5')
        if os.path.isfile(hd5_file):
            dataFrame = pd.read_hdf(hd5_file, 'df_with_missing')

        for src, dst in zip(images, destinations):
            if os.path.isfile(dst) or os.path.islink(dst):
                image_label = scipy.misc.imread(dst, mode='RGB')
                (x_label, y_label, _) = image_label.shape
                label_max_size = max(x_label, y_label)
                if  label_max_size != max_input_size:
                    scale_factor = max_input_size / label_max_size
                    scaled_image = scipy.misc.imresize(image_label, scale_factor)
                    scipy.misc.imsave(dst, scaled_image)
                    first_base = os.path.basename(dst)
                    second_base = os.path.basename(os.path.dirname(dst))
                    image_name = os.path.join('labeled-data',second_base,first_base)
                    dataFrame.loc[str(image_name),:] *= scale_factor

            else:
                image = scipy.misc.imread(src,mode= 'RGB')
                (x,y,_) = image.shape
                max_size = max(x,y)
                scale_factor = max_input_size / max_size
                scaled_image = scipy.misc.imresize(image, scale_factor)

                scipy.misc.imsave(dst, scaled_image)

        if os.path.isfile(hd5_file):
            dataFrame.to_hdf(hd5_file, key='df_with_missing', mode='w')
            dataFrame.to_csv(csv_file)
            pass
    print("\nFrames were selected.\nYou can now label the frames using the function 'label_frames'.")


