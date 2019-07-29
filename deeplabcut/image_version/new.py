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
import os
from pathlib import Path
import shutil
import random
import string


def create_new_project(project, experimenter, images_dir, working_directory=None, copy_images=False, image_type='.png'):
    """Creates a new project directory, sub-directories and a basic configuration file. The configuration file is loaded with the default values. Change its parameters to your projects need.

    Parameters
    ----------
    project : string
        String containing the name of the project.

    experimenter : string
        String containing the name of the experimenter.

    images_dir : list
        A list of string containing the full paths of the images' directory to include in the project.
        Attention: Can also be a file.

    working_directory : string, optional
        The directory where the project will be created. The default is the ``current working directory``; if provided, it must be a string.

    copy_images : bool, optional
        If this is set to True, the images are copied to the ``images`` directory. If it is False,symlink of the images are copied to the project/images directory. The default is ``False``; if provided it must be either
        ``True`` or ``False``.

    image_type : string, optional
        The images'type that will be added to the project.
    Example
    --------
    A full example is provided at test_example file.
    """
    from datetime import datetime as dt
    from deeplabcut.image_version import auxiliaryfunctions
    date = dt.today()
    month = date.strftime("%B")
    day = date.day
    d = str(month[0:3]+str(day))
    date = dt.today().strftime('%Y-%m-%d')
    if working_directory == None:
        working_directory = '.'
    wd = Path(working_directory).resolve()
    project_name = '{pn}-{exp}-{date}'.format(pn=project, exp=experimenter, date=date)
    project_path = wd / project_name

    # Create project and sub-directories
    if project_path.exists():
        print('Project "{}" already exists!'.format(project_path))
        return
    images_path = project_path / 'images'
    data_path = project_path / 'labeled-data'
    shuffles_path = project_path / 'training-datasets'
    results_path = project_path / 'dlc-models'
    for p in [images_path, data_path, shuffles_path, results_path]:
        p.mkdir(parents=True)
        print('Created "{}"'.format(p))

    if isinstance(images_dir,str):
        images_dir = [images_dir]

    images = []
    for image_dir in images_dir:
        if os.path.isdir(image_dir):
            path=image_dir
            images.extend([os.path.join(path,vp) for vp in os.listdir(path) if image_type in vp])
            if len(images)==0:
                print("No images found in",path,os.listdir(path))
                print("Perhaps change the image_type, which is currently set to:", image_type)
            else:
                print("Directory entered, " , len(images)," images were found.")
        else:
            if os.path.isfile(image_dir):
                images.append(image_dir)

    images = [Path(vp) for vp in images]

    # create folder that will have the images in it, so this folder can be used as image set afterwards
    # to be compatible with the deeplabcut code

    temp_folder = ''.join(random.choices(string.ascii_uppercase + string.digits, k=7))
    new_images_path = images_path.joinpath(temp_folder)
    while new_images_path.exists(): # to make sure the temp folder doesn't already exist. If so, generate new temp folder.
        new_images_path = images_path.joinpath(temp_folder)

    new_images_path.mkdir(parents=True, exist_ok=True)
    destinations = [new_images_path.joinpath(vp.name) for vp in images]
    if copy_images==True:
        print("Copying the images")
        for src, dst in zip(images, destinations):
            shutil.copy(os.fspath(src),os.fspath(dst))
    else:
        print("Creating the symbolic link of the image")
        for src, dst in zip(images, destinations):
            if dst.exists(): # and not DEBUG:
                raise FileExistsError('Image {} exists already!'.format(dst))
            try:
                src = str(src)
                dst = str(dst)
                os.symlink(src, dst)
            except OSError:
                import subprocess
                subprocess.check_call('mklink %s %s' %(dst,src),shell = True)
            print('Created the symlink of {} to {}'.format(src, dst))
            images = destinations

    if copy_images==True:
        images=destinations

    images_sets = {}
    try:
        rel_image_path = str(Path.resolve(Path(new_images_path)))
    except:
        rel_image_path = os.readlink(str(new_images_path))
    images_sets[rel_image_path] = {}

    # Check this for your project.
    cfg_file,ruamelFile = auxiliaryfunctions.create_config_template()
    cfg_file
    cfg_file['Task']=project
    cfg_file['scorer']=experimenter
    cfg_file['images_sets']=images_sets
    cfg_file['project_path']=str(project_path)
    cfg_file['date']=d
    cfg_file['bodyparts']=['RightEar','LeftEar','RightEye','LeftEye','Nose']
    cfg_file['cropping']=False
    cfg_file['TrainingFraction']=[0.95]
    cfg_file['iteration']=0
    cfg_file['resnet']=50
    cfg_file['snapshotindex']=-1
    cfg_file['x1']=0
    cfg_file['x2']=640
    cfg_file['y1']=277
    cfg_file['y2']=624
    cfg_file['batch_size']=4
    cfg_file['corner2move2']=(50,50)
    cfg_file['move2corner']=True
    cfg_file['pcutoff']=0.1
    cfg_file['dotsize']=12
    cfg_file['alphavalue']=0.7
    cfg_file['colormap']='jet'

    projconfigfile=os.path.join(str(project_path),'config.yaml')
    auxiliaryfunctions.write_config(projconfigfile,cfg_file)

    print('Generated "{}"'.format(project_path / 'config.yaml'))
    print("\nA new project with name %s is created at %s and a configurable file (config.yaml) is stored there. Change the parameters in this file to adapt to your project's needs.\n Once you have changed the configuration file, use the function 'extract_frames' to select frames for labeling.\n. [OPTIONAL] Use the function 'add_new_videos' to add new videos to your project (at any stage)." %(project_name,str(wd)))
    return projconfigfile

