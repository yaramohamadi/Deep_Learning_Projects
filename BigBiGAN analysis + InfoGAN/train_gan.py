import argparse
import itertools

from src.pipeline import pipeline
from src.training_utils import training_utils
from pathlib import Path
import os 

EXP_HPARAMS = {
    "params": (
        {},
    ),
    "seeds": (420,),
}

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="FMNIST",
                    choices=["FMNIST", "MNIST", "CIFAR10", "CIFAR100", "imagenette", "imagewoof"], help="dataset name")
parser.add_argument("--data_path", type=str, default="../input/fmnist-dataset",
                    help="path to dataset root folder")
parser.add_argument("--model_architecture", type=str, default="bigbigan",
                    choices=["bigbigan", "biggan"], help="type of architecture used in training")
# ADD TRAIN TEST MODE
parser.add_argument("--train", type=str, default="1",
                    choices=["0", "1"], help="0 or 1?")
# ADD LOSS MODE
parser.add_argument("--loss_mode", type=str, default="all",
                    choices=["all", "no_sx", "no_sz", "no_sxz", "info_gan"], help="just pick somethin man")
parser.add_argument("--resume", type=str, default="0", help="which epoch to resume from?")
args = parser.parse_args()


def run_experiments():
    for hparams_overwrite_list, seed in itertools.product(EXP_HPARAMS["params"], EXP_HPARAMS["seeds"]):
        config = training_utils.get_config(args.dataset)
        hparams_str = ""
        for k, v in hparams_overwrite_list.items():
            config[k] = v
            hparams_str += str(k) + "-" + str(v) + "_"
        config["model_architecture"] = args.model_architecture
        config["hparams_str"] = hparams_str.strip("_")
        config["seed"] = seed
        config["loss_mode"] = args.loss_mode
        run_experiment(config)


def run_experiment(config):
    training_utils.set_random_seed(seed=config.seed, device=config.device)

    # Train mode
    if args.train == "1":
      config["train_"] = 1

      # Added option of resuming training
      if args.model_architecture == "bigbigan" and args.resume == "0":
          training_pipeline = pipeline.BigBiGANPipeline.from_config(data_path=args.data_path, config=config)

      elif args.model_architecture == "bigbigan" and args.resume != "0":
          print("Loading data from checkpoint, resuming from epoch: " + args.resume)
          checkpoint_path="/content/drive/MyDrive/BigBiGAN-PyTorch-main/data/MNIST/bigbigan/"  + args.loss_mode + "/checkpoints/checkpoint.pth"
          training_pipeline = pipeline.BigBiGANPipeline.from_checkpoint(data_path=args.data_path, 
                                                                        checkpoint_path = checkpoint_path, 
                                                                        config=config)
      
      else:
          raise ValueError(f"Architecture type {args.model_architecture} is not supported")
      training_pipeline.train_model(resume=int(args.resume))

    # TEST MODE
    else:
      config["train_"] = 0
      test_pipeline = pipeline.BigBiGANInference.from_checkpoint(data_path=args.data_path, 
          checkpoint_path="/content/drive/MyDrive/BigBiGAN-PyTorch-main/data/MNIST/bigbigan/" + args.loss_mode + "/checkpoints/checkpoint.pth",  
          config=config)
      # InfoGAN: Save images for each C class ################
      if args.loss_mode == 'info_gan':
        print("creating C gen directory for InfoGAN")
        test_pipeline.save_img()
      ########################################################
      # Loss modes: create FID folders and calculate IS too ##
      if not os.path.exists(config.save_org_path):
        print("creating original image directory (save one batch")
        test_pipeline.create_FID_path_original()
      print("original images saved")
      if True:
        print("creating generated image directory (save one batch")
        test_pipeline.create_FID_path_generated()
      print("generated images saved")
      # Save encoded train and test images
      if True:
        print("creating encoded test csv file")
        # Save test encoded 
        test_pipeline.save_encoded()
      print("encoded test csv saved")
      if True:
        print("creating encoded train csv file")
        # Save train endoced
        config["train_"] =  1
        test_pipeline = pipeline.BigBiGANInference.from_checkpoint(data_path=args.data_path, 
              checkpoint_path="/content/drive/MyDrive/BigBiGAN-PyTorch-main/data/MNIST/bigbigan/" + args.loss_mode + "/checkpoints/checkpoint.pth",  
              config=config)
        test_pipeline.save_encoded()
      print("encoded train csv saved")
      ########################################################

run_experiments()
