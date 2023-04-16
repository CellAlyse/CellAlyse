import os
import utils.dirtools

config_vars = {}

config_vars["root_directory"] = 'models'

config_vars["max_training_images"] = 0

config_vars["create_split_files"] = False

config_vars["training_fraction"] = 0.5
config_vars["validation_fraction"] = 0.25

config_vars["transform_images_to_PNG"] = True
config_vars["pixel_depth"] = 8

config_vars["min_nucleus_size"] = 4

config_vars["boundary_size"] = 2

config_vars["augment_images"] =  False

config_vars["elastic_points"] = 16
config_vars["elastic_distortion"] = 5

config_vars["elastic_augmentations"] = 10

config_vars["learning_rate"] = 1e-4

config_vars["epochs"] = 3

config_vars["steps_per_epoch"] = 300

config_vars["batch_size"] = 10

config_vars["val_batch_size"] = 1

config_vars["rescale_labels"] = True

config_vars["crop_size"] = 128

config_vars["cell_min_size"] = 4

config_vars["boundary_boost_factor"] = 1

config_vars["object_dilation"] = 3
config_vars["MIN_SIZE"] = 3
config_vars = utils.dirtools.setup_working_directories(config_vars)

