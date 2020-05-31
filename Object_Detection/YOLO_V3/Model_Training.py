"""
This Script will help in Training the YOLOv3 Model on the custom dataset

It will take Prepared Data as Argument.

MODIFIED FROM keras-yolo3 PACKAGE, https://github.com/qqwweee/keras-yolo3
Retrain the YOLO model for your own dataset.

Author: Tushar Goel

"""

import os
import sys

#Importing Models 
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import (
    TensorBoard,
    ModelCheckpoint,
    ReduceLROnPlateau,
    EarlyStopping,
)
from yolo3.model import (
    preprocess_true_boxes,
    yolo_body,
    tiny_yolo_body,
    yolo_loss,
)

from yolo3.utils import get_random_data
from PIL import Image
from time import time
import pickle

from Train_Utils import (
    get_classes,
    get_anchors,
    create_model,
    create_tiny_model,
    data_generator,
    data_generator_wrapper,
    ChangeToOtherMachine,
)

def Train_Yolo(working_directory,val_split = 0.1,is_tiny=False,random_seed = None,epochs = 51,batch_size1 = 32,batch_size2 = 4):
    """
    This Function will be use to Train user their own Model for custom data 

    """
    
    YOLO_filename = os.path.join(working_directory,'data_train.txt')
    YOLO_classname = os.path.join(working_directory,'data_classes.txt')
    anchors_path = 'yolo_anchors.txt'
    weights_path = os.path.join(working_directory,'yolo.h5')
    log_dir = working_directory
    
    np.random.seed(random_seed)
    class_name = get_classes(YOLO_classname)
    num_classes = len(class_name)
    anchors = get_anchors(anchors_path)
    
    input_shape = (416,416)
    
    is_tiny_version = len(anchors) == 6
    epoch1, epoch2 = epochs,epochs
    if is_tiny:
        model = create_tiny_model(
            input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path
        )
    else:
        model = create_model(
            input_shape, anchors, num_classes, freeze_body=2, weights_path=weights_path
        )  # make sure you know what you freeze
    
    
    log_dir_time = os.path.join(log_dir, "{}".format(int(time())))
    logging = TensorBoard(log_dir=log_dir_time)
    checkpoint = ModelCheckpoint(
        os.path.join(log_dir, "checkpoint.h5"),
        monitor="val_loss",
        save_weights_only=True,
        save_best_only=True,
        period=5,
    )
    reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=10, verbose=1
    )

    val_split = val_split
    with open(YOLO_filename) as f:
        lines = f.readlines()
    
    
    # This step makes sure that the path names correspond to the local machine
    # This is important if annotation and training are done on different machines (e.g. training on AWS)
    #lines = ChangeToOtherMachine(lines, remote_machine="")
    np.random.shuffle(lines)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val
    
    # Train with frozen layers first, to get a stable loss.
    # Adjust num epochs to your dataset. This step is enough to obtain a decent model.
    if True:
        model.compile(
            optimizer=Adam(lr=1e-3),
            loss={
                # use custom yolo_loss Lambda layer.
                "yolo_loss": lambda y_true, y_pred: y_pred
            },
        )

        batch_size = batch_size1
        print(
            "Train on {} samples, val on {} samples, with batch size {}.".format(
                num_train, num_val, batch_size
            )
        )
        history = model.fit_generator(
            data_generator_wrapper(
                lines[:num_train], batch_size, input_shape, anchors, num_classes
            ),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(
                lines[num_train:], batch_size, input_shape, anchors, num_classes
            ),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch1,
            initial_epoch=0,
            callbacks=[logging, checkpoint],
        )
        model.save(os.path.join(log_dir, "trained_weights_stage_1.h5"))

        step1_train_loss = history.history["loss"]

        file = open(os.path.join(log_dir_time, "step1_loss.npy"), "w")
        with open(os.path.join(log_dir_time, "step1_loss.npy"), "w") as f:
            for item in step1_train_loss:
                f.write("%s\n" % item)
        file.close()

        step1_val_loss = np.array(history.history["val_loss"])

        file = open(os.path.join(log_dir_time, "step1_val_loss.npy"), "w")
        with open(os.path.join(log_dir_time, "step1_val_loss.npy"), "w") as f:
            for item in step1_val_loss:
                f.write("%s\n" % item)
        file.close()

    # Unfreeze and continue training, to fine-tune.
    # Train longer if the result is unsatisfactory.
    if True:
        for i in range(len(model.layers)):
            model.layers[i].trainable = True
        model.compile(
            optimizer=Adam(lr=1e-4), loss={"yolo_loss": lambda y_true, y_pred: y_pred}
        )  # recompile to apply the change
        print("Unfreeze all layers.")

        batch_size = (
            batch_size2  # note that more GPU memory is required after unfreezing the body
        )
        print(
            "Train on {} samples, val on {} samples, with batch size {}.".format(
                num_train, num_val, batch_size
            )
        )
        history = model.fit_generator(
            data_generator_wrapper(
                lines[:num_train], batch_size, input_shape, anchors, num_classes
            ),
            steps_per_epoch=max(1, num_train // batch_size),
            validation_data=data_generator_wrapper(
                lines[num_train:], batch_size, input_shape, anchors, num_classes
            ),
            validation_steps=max(1, num_val // batch_size),
            epochs=epoch1 + epoch2,
            initial_epoch=epoch1,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping],
        )
        model.save(os.path.join(log_dir, "trained_weights_final.h5"))
        step2_train_loss = history.history["loss"]

        file = open(os.path.join(log_dir_time, "step2_loss.npy"), "w")
        with open(os.path.join(log_dir_time, "step2_loss.npy"), "w") as f:
            for item in step2_train_loss:
                f.write("%s\n" % item)
        file.close()

        step2_val_loss = np.array(history.history["val_loss"])

        file = open(os.path.join(log_dir_time, "step2_val_loss.npy"), "w")
        with open(os.path.join(log_dir_time, "step2_val_loss.npy"), "w") as f:
            for item in step2_val_loss:
                f.write("%s\n" % item)
        file.close()

    return history












