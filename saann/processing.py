# processing.py
# Copyright (c) 2026 Alessio Branda
# Licensed under the MIT License

import pandas as pd
import warnings
import matplotlib.image as mpimg
from PIL import Image
import sys
import os
from os import listdir
from os.path import isfile, join, isdir
from . import backend as BE


def train_test_split(X, y, split_test_percentage = 0.3):
    """
    Returns the X_train, X_test, y_train and y_test separated in regards to the split specified.
    
    Parameters
    ----------
    :param X: Features/input Ndarray.
    :param y: Target Ndarray.
    :param split_test_percentage: Size of the test array in percentage (0, 1) of the total size.
    
    Returns
    -------
    :return X_train: Array of inputs for training.
    :return X_test: Array of inputs for testing.
    :return y_train: Array of targets for training.
    :return y_test: Array of targets for testing.
    
    Raises
    ------
    :ValueError: If the split test percentage is not a float between 0 and 1, inclusive.\n
    :ValueError: If param X's (or y's) type is not in (list, BE.xp.ndarray, pd.DataFrame).\n
    :ValueError: If param X and param y have different lengths.
    
    Example
    -------
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y, split_test_percentage = 0.3)
    """
    if isinstance(X, (list, BE.xp.ndarray, pd.DataFrame)):
        if isinstance(X, BE.xp.ndarray):
            pass
        else:
            try:
                tmp = type(X)
                X = BE.xp.array(X)
                warnings.warn(f"Converted 'X' ({tmp}) to {type(X)}.")
            except:
                raise TypeError(f"The given 'X' type ({type(X)}) is not supported.")
    else:
        raise TypeError(f"The given 'X' type ({type(X)}) is not supported.")
    
    if isinstance(y, (list, BE.xp.ndarray, pd.DataFrame)):
        if isinstance(y, BE.xp.ndarray): 
            pass
        else:
            try:
                tmp = type(y)
                y = BE.xp.array(y)
                warnings.warn(f"Converted 'y' ({tmp}) to {type(y)}.")
            except:
                raise TypeError(f"The given 'y' type ({type(y)}) is not supported.")
    else:
        raise TypeError(f"The given 'y' type ({type(y)}) is not supported.")
    if len(X) != len(y):
        raise ValueError(f"Lengths of X ({len(X)}) and y ({len(y)}) arrays do not match.")
    if split_test_percentage <= 0 or split_test_percentage >= 1:
        raise ValueError(f"The 'split_test_percentage' needs to be in the range (0, 1).")
    if split_test_percentage >= 0.5:
        warnings.warn("It is reccomended to use a 'split_test_percentage' of around 0.3.")

    X_train = X[0:int((1-split_test_percentage)*len(X))]
    y_train = y[0:int((1-split_test_percentage)*len(X))]
    X_test = X[int((1-split_test_percentage)*len(X)):]
    y_test = y[int((1-split_test_percentage)*len(X)):]

    try:
        y_train = y_train.reshape(-y.shape[1], y.shape[1])
        y_test = y_test.reshape(-y.shape[1], y.shape[1])
    except:
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    return X_train, X_test, y_train, y_test

class Scaling:
    """
    Class for the different scalings
    """

    def __init__(self):
        self.scaled_data = None

    def zScore(self, x):
        """
        Performs a z-score standardization.\n
        Parameter
        ---------
        :param x: Data array

        Return
        ------
        :return scaled_data: Data scaled via z-score standardization

        Example
        -------
            >>> scaler = Scaling()
        >>> x_scaled = scaler.zScore(x)
        """
        self.scaled_data = (x-BE.xp.mean(x))/(BE.xp.std(x))
        return self.scaled_data
    
    def MinMax(self, x):
        """
        Performs a Min-Max standardization.\n
        Parameters
        ----------
        :param x: Data array

        Return
        ------
        :return scaled_data: Data scaled via Min-Max standardization

        Example
        -------
            >>> scaler = Scaling()
        >>> x_scaled = scaler.MinMax(x)
        """
        self.scaled_data = (x - BE.xp.min(x))/(BE.xp.max(x)-BE.xp.min(x))
        return self.scaled_data
    
    def LogNorm(self, x):
        """
        Performs a natural log transformation.\n
        Parameters
        ----------
        :param x: Data array

        Return
        ------
        :return scaled_data: Data scaled via natural log normalization

        Example
        -------
            >>> scaler = Scaling()
        >>> x_scaled = scaler.LogNorm(x)
        """
        self.scaled_data = BE.xp.log1p(x)
        return self.scaled_data
    
    def MeanNorm(self, x):
        """
        Performs a mean normalization.\n
        Parameters
        ----------
        :param x: Data array

        Return
        ------
        :return scaled_data: Data scaled via mean normalization

        Example
        -------
            >>> scaler = Scaling()
        >>> x_scaled = scaler.MeanNorm(x)
        """
        self.scaled_data = (x-BE.xp.mean(x))/(BE.xp.max(x)-BE.xp.min(x))
        return self.scaled_data

class ImageProcessing:

    def __init__(self, images_path):
        self.path = images_path
        self.X = []
        self.y = []

    def image_upload(self, rel_path = None):
        if rel_path == None:
            rel_path = self.res_path
    
        images = [f for f in listdir(rel_path) if isfile(join(rel_path, f))]
        idx = 0
        for image in images:
            tag = image.split(sep="_")[0]
            if sys.platform.startswith("win"):
                raw = mpimg.imread(rf"{rel_path}\{image}")
            else:
                raw = mpimg.imread(rf"{rel_path}/{image}")
            self.X.append(raw)
            self.y.append(tag)
            if (idx+1) % 100 == 0:
                print("Processing image #",idx+1)
            idx += 1
        return self.X, self.y
    
    def prepare_images(self, size = 128, amount = None):
        self.num_images = amount
        list_dir = [f for f in listdir(self.path) if isdir(join(self.path, f))] 

        if "resized" in list_dir: # if ran multiple times
            list_dir.remove("resized")

        try:
            list_dir.remove("resized_tmp")
        except:
            pass
            #print("Removed the 'resized' directory from the list.")

        print("List of directories: ", list_dir)

        self.list_dir = list_dir.copy()

        for dir in list_dir:
            if sys.platform.startswith("win"):
                rel_path = rf"{self.path}\{dir}"
            else:
                rel_path = rf"{self.path}/{dir}"

            resized_path = self.resize_images(rel_path=rel_path, og_path=self.path, size=size)

        self.res_path = resized_path
                
    
    def resize_images(self, rel_path, og_path, size):
        images = [f for f in listdir(rel_path) if isfile(join(rel_path, f))]
        idx = 0
        try:
            tag = rel_path.split(sep="\\")[-1]
        except:
            tag = rel_path.split(sep="/")[-1]
        min_size = (size, size)
        idx = 1

        if sys.platform.startswith("win"): #on Windows systems
            tag = rel_path.split(sep="\\")[-1]

            try:
                os.mkdir(rf"{og_path}\resized")
            except:
                #print(fr"Directory {og_path}\resized already exists.")
                pass

            for image in images:
                img = Image.open(rf"{rel_path}\{image}")
                img = img.resize(min_size)
                img.save(rf"{og_path}\resized\{tag}_{idx}.jpg", optimize=True, quality=70)

                if isinstance(self.num_images, int) and idx == self.num_images:
                    break

                idx += 1
            return rf"{og_path}\resized"

        else: # on Unix-like systems
            tag = rel_path.split(sep="/")[-1]

            try:
                os.mkdir(rf"{og_path}/resized")
            except:
                print(fr"Directory {og_path}/resized already exists.")

            for image in images:
                img = Image.open(rf"{rel_path}/{image}")
                img = img.resize(min_size)
                img.save(rf"{og_path}/resized/{tag}_{idx}.jpg", optimize=True, quality=70)

                if isinstance(self.num_images, int) and idx == self.num_images:
                    break

                idx += 1
            return rf"{og_path}/resized"
        
    def clean_datset(self, X = None, y = None):
        if X == None:
            X = self.X
        if y == None:
            y = self.y

        X_clean = []
        y_clean = []

        self.clean_flag = False

        for array, target in zip(X, y):
            if len(array.shape) < 3:
                continue
            if array.shape[2] == 4:
                X_clean.append(BE.xp.delete(array, obj = 3, axis=2))
                y_clean.append(target)
                self.clean_flag = True
            else:
                X_clean.append(array)
                y_clean.append(target)
        
        if self.clean_flag:
            print("Dataset has been cleaned.")
        else:
            print("Dataset already clean.")
        
        self.X = X_clean.copy()
        self.y = y_clean.copy()

        return self.X, self.y
    
    def prepare_features(self, X = None):
        if X == None:
            X = self.X

        X = BE.xp.array([BE.xp.asarray(x) for x in X])
        self.X = (X.astype('float32') / 255.0) - 0.5

        return self.X
    
    def prepare_targets(self, y = None, list_classes = None):
        if y == None:
            y = self.y
        
        if list_classes == None:
            list_classes = self.list_dir

        list_classes = [l.split(sep="_")[0] for l in list_classes]
    
        y_tmp = BE.xp.zeros((len(y), len(list_classes)))

        for i in range(len(y)):
            idx = list_classes.index(y[i])
            y_tmp[i][idx] = 1

        self.y = y_tmp.copy()

        self.list_classes = list_classes

        return self.y
    
    def get_classes(self):
        try:
            return self.list_classes
        except:
            warnings.warn("No classes found.")

    def shuffle_dataset(self, X = None, y = None):
        if X == None and y != None or X != None and y == None:
            raise ValueError("Please, either provide both arrays or neither.")
        if X == None:
            X = self.X
        if y == None:
            y = self.y
        if len(X) != len(y):
            raise ValueError(f"Lengths of the feature array and target array do not match: {len(X)} - {len(y)}")

        perm = BE.xp.random.permutation(len(X))
        self.X = X[perm]
        self.y = y[perm]

        print("Dataset has been shuffled.")

        return self.X, self.y
    
    def ready_dataset(self, size, amount = None, remove_resized = False, shuffle = True, split_test_percentage = None):
        self.prepare_images(size=size, amount=amount)
        self.image_upload()
        self.clean_datset()
        self.prepare_features()
        self.prepare_targets()
        if shuffle:
            self.shuffle_dataset()
        if remove_resized:
            import shutil
            try:
                shutil.rmtree(self.res_path, ignore_errors=True)
                print(rf"{self.res_path} has been successfully deleted.")
            except ImportError as e:
                print(rf"{self.res_path} has not been been deleted: {e}.")

        if split_test_percentage == None:
            print("Dataset is ready to use")
            return self.X, self.y, self.list_classes
        else:
            X_train, X_test, y_train, y_test = train_test_split(X = self.X, y = self.y, split_test_percentage=split_test_percentage)
            print(f"Dataset is ready and has been split into Train (length = {len(X_train)}) and Test (length = {len(X_test)}) sets.")
            return X_train, X_test, y_train, y_test, self.list_classes
