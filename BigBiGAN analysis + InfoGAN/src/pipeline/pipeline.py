import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy
import torch
import torch.nn as nn
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3

from tqdm import tqdm
from pathlib import Path
import os 

from src.data_processing import data_loading
from src.pipeline import logger as training_logger
from src.model import architecture
from src.model import losses


class Pipeline:
    def __init__(
            self, dataloader, model, gen_criterion, disc_criterion,
            gen_optimizer, disc_optimizer, logger, config
    ):
        self.dataloader = dataloader
        self.model = model
        self.gen_criterion = gen_criterion
        self.disc_criterion = disc_criterion
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.logger = logger
        self.config = config
        self.counter = 0

    # Loss modes: add starting epoch option so we can keep training loaded model
    def train_model(self, resume=0):
        for epoch in range(self.config.epochs):
            self.counter = 0
            # resume option added
            self.run_epoch(epoch + resume)

    def save_model(self, epoch):
        if (epoch % self.config.save_model_interval == 0) and epoch:
            save_folder = Path(self.config.save_model_path.format(
                    ds_name=self.config.ds_name,
                    model_architecture=self.config.model_architecture,
                    loss_mode=self.config.loss_mode,
                    hparams=self.config.hparams_str,
            ))
            save_folder.mkdir(parents=True, exist_ok=True)
            save_path = os.path.join(save_folder, "checkpoint.pth")
            torch.save(self.model.state_dict(), save_path)

    # InfoGAN: Add c_info #########
    def save_img(self, epoch, real_img, img_gen, latent=None, y=None, c_info=None):
    ###############################
        if epoch % self.config.save_metric_interval == 0 and self.counter == 0:
            with torch.no_grad():
                fake = img_gen.detach().cpu()[:self.config.save_img_count, ...]
            fake_img = np.transpose(vutils.make_grid(
                fake, padding=2, nrow=self.config.img_rows, normalize=True), (1, 2, 0))
            plt.imshow(fake_img)

            file_name = f"ep{epoch}_step{self.counter}.png"
            gen_imgs_save_folder = Path(self.config.gen_imgs_save_path.format(
                ds_name=self.config.ds_name,
                model_architecture=self.config.model_architecture,
                loss_mode=self.config.loss_mode,
                hparams=self.config.hparams_str,
            ))
            gen_imgs_save_folder.mkdir(parents=True, exist_ok=True)
            gen_imgs_save_path = str(gen_imgs_save_folder / file_name)
            plt.savefig(fname=gen_imgs_save_path)

            if latent is not None:
                img_gen, noise = self.model.generate_imgs(cls=y, noise=latent, c_info=c_info)
                img_gen = img_gen.detach().cpu()[:self.config.save_img_count, ...]
                img_gen = np.transpose(vutils.make_grid(
                    img_gen, padding=2, nrow=self.config.img_rows, normalize=True), (1, 2, 0))
                plt.imshow(img_gen)

                file_name = f"ep{epoch}_step{self.counter}_reconstructed.png"
                gen_imgs_save_folder = Path(self.config.gen_imgs_save_path.format(
                    ds_name=self.config.ds_name,
                    model_architecture=self.config.model_architecture,
                    loss_mode=self.config.loss_mode,
                    hparams=self.config.hparams_str,
                ))
                gen_imgs_save_folder.mkdir(parents=True, exist_ok=True)
                gen_imgs_save_path = str(gen_imgs_save_folder / file_name)
                plt.savefig(fname=gen_imgs_save_path)

        self.counter += 1


class BigBiGANPipeline(Pipeline):
    def run_epoch(self, epoch):
        for step, (x, y) in tqdm(enumerate(self.dataloader)):
            x, y = x.to(device=self.config.device), y.to(device=self.config.device)

            # InfoGAN: create C as random uniform onehot vector size 10 ######################
            c_info = nn.functional.one_hot(y, num_classes=10).float()
            c_info = c_info.to(device=self.config.device)

            self.model.req_grad_disc(True)
            for _ in range(self.config.disc_steps):
                # Add C as input to generator
                img_gen, noise = self.model.generate_imgs(cls=y, c_info=c_info)
                ##############################################################################
                z_img = self.model.generate_latent(img=x)
                self.disc_optimizer.zero_grad()
                # InfoGAN: Output contains predicted C too!
                outputs = self.model.forward(
                    img_real=x,
                    img_gen=img_gen.detach(),
                    z_noise=noise,
                    z_img=z_img.detach(),
                    cls=y
                )
                # InfoGAN: input c_info and generated c (part of generated latent) to discriminator loss #######
                c_info_gen = z_img.detach()[:, :10]
                disc_loss = self.disc_criterion(outputs, c_info, c_info_gen)
                ################################################################################################
                # Some of the weights are needed for generator too
                disc_loss.backward()
                self.disc_optimizer.step()
            self.model.req_grad_disc(False)
            self.gen_optimizer.zero_grad()
            outputs = self.model.forward(img_real=x, img_gen=img_gen, z_noise=noise, z_img=z_img, cls=y)
            # InfoGAN: input c_info and generated c (part of generated latent) to generator loss ###########
            gen_enc_loss = self.gen_criterion(outputs, c_info, c_info_gen)
            ################################################################################################
            gen_enc_loss.backward()
            self.gen_optimizer.step()

            # InfoGAN: Add c_info to save_img ######
            self.save_img(epoch, x, img_gen, z_img, y, c_info)
            ########################################
            self.save_model(epoch)
            self.logger(epoch, step, disc_loss, gen_enc_loss)

    @classmethod
    def from_config(cls, data_path, config):
        print("creating model from configuration")
        config.device = torch.device(config.device)
        dataloader = data_loading.get_supported_loader(config.ds_name)(data_path, config)
        model = architecture.BigBiGAN.from_config(config).to(device=config.device)

        # Add loss modes
        gen_enc_criterion = losses.GeneratorEncoderLoss(loss_mode=config.loss_mode)
        # Add loss modes
        disc_criterion = losses.BiDiscriminatorLoss(loss_mode=config.loss_mode)
        
        gen_enc_optimizer = torch.optim.Adam(model.get_gen_enc_params(), lr=config.lr_gen, betas=config.betas)
        disc_optimizer = torch.optim.Adam(model.get_disc_params(), lr=config.lr_disc, betas=config.betas)
        
        logger = training_logger.BiGANLogger.from_config(config=config, name=config.hparams_str)
        return cls(
            model=model,
            gen_criterion=gen_enc_criterion,
            disc_criterion=disc_criterion,
            gen_optimizer=gen_enc_optimizer,
            disc_optimizer=disc_optimizer,
            dataloader=dataloader,
            logger=logger,
            config=config
        )
    # Loss mode : add this so we can keep training from checkpoint #######
    @classmethod
    def from_checkpoint(cls, data_path, checkpoint_path, config):
        print("creating model from checkpoint")
        dataloader = data_loading.get_supported_loader(config.ds_name)(data_path, config)
        model = architecture.BigBiGAN.from_config(config).to(device=config.device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=True)
        model = model.cuda()
        model = model.train()
        gen_enc_criterion = losses.GeneratorEncoderLoss(loss_mode=config.loss_mode)
        disc_criterion = losses.BiDiscriminatorLoss(loss_mode=config.loss_mode)
        gen_enc_optimizer = torch.optim.Adam(model.get_gen_enc_params(), lr=config.lr_gen, betas=config.betas)
        disc_optimizer = torch.optim.Adam(model.get_disc_params(), lr=config.lr_disc, betas=config.betas)
        logger = training_logger.BiGANLogger.from_config(config=config, name=config.hparams_str)
        return cls(
            model=model,
            gen_criterion=gen_enc_criterion,
            disc_criterion=disc_criterion,
            gen_optimizer=gen_enc_optimizer,
            disc_optimizer=disc_optimizer,
            dataloader=dataloader,
            logger=logger,
            config=config)
    ######################################################################

class BigBiGANInference:
    def __init__(self, model, dataloader, config):
        self.model = model
        self.dataloader = dataloader
        self.config = config

    # Loss modes: Save input images ##############################
    def create_FID_path_original(self):
        batch_num_to_save = 1
        save_org_path = Path(self.config.save_org_path.format(
          ds_name=self.config.ds_name,
          model_architecture=self.config.model_architecture))
        save_org_count = 0
        save_org_path.mkdir(parents=True, exist_ok=True)
        for step, (org_img, y) in tqdm(enumerate(self.dataloader)):
          if step == batch_num_to_save:
            print('done')
            break;
          for im in tqdm(org_img):
            file_name = f"{save_org_count}.png"
            plt.imshow(im[0])
            plt.axis('off')
            file_path = os.path.join(save_org_path, file_name)
            plt.savefig(fname=file_path, bbox_inches='tight',transparent=True, pad_inches=0)
            save_org_count += 1
    #############################################################

    # Inception Score calculation #############################
    def inception_score(self, gen_img):
         
          # Load inception model
          inception_model = inception_v3(pretrained=True, transform_input=False)
          inception_model.eval();
          
          # Size 75 75 so computation is lighter
          up = nn.Upsample(size=(75, 75), mode='bilinear')
          def get_pred(x):
              x = up(x)
              x = inception_model(x)
              return F.softmax(x).data.cpu().numpy()

          # Get input ready (image has one channels but inception accepts RGB)
          inception_input = torch.cat(3*[gen_img], axis=1)

          predictions = get_pred(inception_input)

          split_scores = []
          # Now compute the mean kl-div
          py = np.mean(predictions, axis=0)
          scores = []
          for i in range(256):
              pyx = predictions[i]
              scores.append(entropy(pyx, py))
          split_scores = np.exp(np.mean(scores))

          print("___________________________________")
          print("Inception Score is :::::: ", np.mean(split_scores))
          print("___________________________________")
    ###########################################################

    # Loss modes: Save generated images #########################
    def create_FID_path_generated(self):
        batch_num_to_save = 1
        save_gen_path = Path(self.config.save_gen_path.format(
          ds_name=self.config.ds_name,
          loss_mode=self.config.loss_mode,
          model_architecture=self.config.model_architecture))
        save_gen_count = 0
        save_gen_path.mkdir(parents=True, exist_ok=True)
        for step, (org_img, y) in enumerate(self.dataloader):
          if step == batch_num_to_save:
            print('done')
            break;

          # Generate batch of fake images
          org_img, y = org_img.to(device=self.config.device), y.to(device=self.config.device)
          latent = self.encode(org_img)
          gen_img = self.generate(y, latent)[0]
          gen_img = gen_img.detach().cpu()

          # Calculate Inception Score:
          self.inception_score(gen_img)

          # Save batch of images
          gen_img = gen_img.detach().cpu()
          for im in tqdm(gen_img):
            file_name = f"{save_gen_count}.png"
            plt.imshow(im[0])
            plt.axis('off')
            file_path = os.path.join(save_gen_path, file_name)
            plt.savefig(fname=file_path, bbox_inches='tight',transparent=True, pad_inches=0)
            save_gen_count += 1
    ###########################################################

    # Loss modes: Save encoded ################################
    def save_encoded(self):
        # Save labels and encoded vector in this
        DF = pd.DataFrame(columns=range(101))
        for step, (org_img, y) in tqdm(enumerate(self.dataloader)):
            # moved data to GPU
            org_img, y = org_img.to(device=self.config.device), y.to(device=self.config.device)
            latent = self.encode(org_img)

            # Add labels and encoded vector to dataframe
            encoded = latent.detach().cpu()
            labels = y.detach().cpu().reshape(-1, 1)
            encoded = np.concatenate((labels, encoded), axis = 1)
            DF = pd.concat([DF, pd.DataFrame(encoded)], axis=0)

        save_encoded_path = Path(self.config.save_encoded_path.format(
          ds_name=self.config.ds_name,
          loss_mode=self.config.loss_mode,
          model_architecture=self.config.model_architecture))
        save_gen_count = 0
        save_encoded_path.mkdir(parents=True, exist_ok=True)
        if self.config["train_"] == 1:
          DF.to_csv(os.path.join(save_encoded_path, 'train.csv'), index=False)
        else:
          DF.to_csv(os.path.join(save_encoded_path, 'test.csv'), index=False)
        print("saved encoded")
    #########################################################

    def inference(self):
        for step, (org_img, y) in tqdm(enumerate(self.dataloader)):
            latent = self.encode(org_img)
            reconstructed_img = self.generate(y, latent)
            self.save_img(org_img, reconstructed_img)
            break


    def encode(self, img):
        z_img = self.model.generate_latent(img=img)
        return z_img

    # InfoGAN add C to generator ###################
    def generate(self, y, latent, c_info=None):
        if c_info == None:
          c_info = nn.functional.one_hot(y, num_classes=10).float()
          c_info = c_info.to(device=self.config.device)
        img_gen, noise = self.model.generate_imgs(cls=y, noise=latent, c_info=c_info)
        return img_gen, noise
    ########################

    # INFOGAN save images for each C ######################################
    def save_img(self):

        if self.config.loss_mode == 'info_gan':
          batch_num_to_save = 1

          save_gen_path = Path(self.config.save_gen_c_path.format(
            ds_name=self.config.ds_name,
            loss_mode=self.config.loss_mode,
            model_architecture=self.config.model_architecture))
          save_gen_path.mkdir(parents=True, exist_ok=True)

          for step, (org_img, y) in tqdm(enumerate(self.dataloader)):
            if step == batch_num_to_save:
              print('done')
              break;
            org_img, y = org_img.to(device=self.config.device), y.to(device=self.config.device)
            latent = self.encode(org_img)

            indices = torch.tensor(range(10))
            c_info_choices = nn.functional.one_hot(y, num_classes=10).float()

            # For each C variable save 100 images
            for i, c in enumerate(c_info_choices.numpy()):
              # Create a batch of a C choice
              c_info = torch.tensor([c])
              c_info = torch.cat(256*[c_info]).to(device=self.config.device)

              gen_img = self.generate(y, latent, c_info)[0]
              # Choose 100 images to save
              gen_img = gen_img.detach().cpu()[:100]
              img = np.transpose(vutils.make_grid(
                  gen_img, padding=2, nrow=10, normalize=True), (1, 2, 0)) * 255
              plt.imshow(img)

              file_name = f"{i}.png"
              gen_imgs_save_path = str(save_gen_path / file_name)
              plt.savefig(fname=gen_imgs_save_path)
    ##########################################################################

    @classmethod
    def from_checkpoint(cls, data_path, checkpoint_path, config):
        dataloader = data_loading.get_supported_loader(config.ds_name)(data_path, config)
        model = architecture.BigBiGAN.from_config(config).to(device=config.device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint, strict=True)
        model = model.cuda()
        model = model.eval()
        return cls(model=model, dataloader=dataloader, config=config)


class GANPipeline(Pipeline):
    def run_epoch(self, epoch):
        for step, (x, y) in tqdm(enumerate(self.dataloader)):
            x, y = x.to(device=self.config.device), y.to(device=self.config.device)
            if self.model.cls is None: self.model.cls = y.detach()
            img_gen, noise = self.model.generate_imgs(cls=y)

            self.model.req_grad_disc(True)
            disc_loss, disc_real_acc, disc_fake_acc = self.forward_disc(x, img_gen, y)
            self.model.req_grad_disc(False)
            gen_loss, gen_disc_acc = self.forward_gen(img_gen, y, noise)

            self.save_img(epoch, x, img_gen)
            if (epoch % self.config.save_model_interval == 0) and epoch:
                torch.save(self.model.state_dict(), self.config.save_model_path)
            self.logger(epoch, step, disc_loss, gen_loss, gen_disc_acc, disc_real_acc, disc_fake_acc)

    def forward_gen(self, gen_img, y, noise):
        for i in range(self.config.gen_steps):
            self.model.generator.zero_grad()
            _, pred_gen_img = self.model.discriminator(x=gen_img, cls=y)
            pred_gen_img = torch.sigmoid(pred_gen_img.reshape(-1))

            label_gen_img = torch.ones(pred_gen_img.shape[0], device=self.config.device)
            gen_loss = self.gen_criterion(pred_gen_img, label_gen_img)
            gen_loss.backward()

            gen_disc_acc = 1 - pred_gen_img.mean().item()
            self.gen_optimizer.step()

            if self.config.gen_steps > 1:
                gen_img, _ = self.model.generate_imgs(cls=y, noise=noise)

        return gen_loss, gen_disc_acc

    def forward_disc(self, img, gen_img, y):
        for _ in range(self.config.disc_steps):
            self.model.discriminator.zero_grad()
            _, pred_real_img = self.model.discriminator(x=img, cls=y)
            pred_real_img = torch.sigmoid(pred_real_img.reshape(-1))

            label_real_img = torch.ones(pred_real_img.shape[0], device=self.config.device)
            real_img_loss = self.disc_criterion(pred_real_img, label_real_img)
            real_img_loss.backward()

            _, pred_gen_img = self.model.discriminator(x=gen_img.detach(), cls=y)
            pred_gen_img = torch.sigmoid(pred_gen_img.reshape(-1))

            label_gen_img = torch.zeros(pred_gen_img.shape[0], device=self.config.device)
            gen_img_loss = self.disc_criterion(pred_gen_img, label_gen_img)
            gen_img_loss.backward()

            disc_real_acc = pred_real_img.mean().item()
            disc_fake_acc = 1 - pred_gen_img.mean().item()

            disc_loss = gen_img_loss + real_img_loss

            self.disc_optimizer.step()

        return disc_loss, disc_real_acc, disc_fake_acc

    @classmethod
    def from_config(cls, data_path, config):
        config.device = torch.device(config.device)
        dataloader = data_loading.get_supported_loader(config.ds_name)(data_path, config)
        model = architecture.BigGAN.from_config(config).to(device=config.device)

        gen_criterion = torch.nn.BCELoss()
        disc_criterion = torch.nn.BCELoss()

        gen_optimizer = torch.optim.Adam(model.get_gen_params(), lr=config.lr_gen, betas=config.betas)
        disc_optimizer = torch.optim.Adam(model.get_disc_params(), lr=config.lr_disc, betas=config.betas)

        logger = training_logger.GANLogger.from_config(config=config, name=config.hparams_str)
        return cls(
            model=model,
            gen_criterion=gen_criterion,
            disc_criterion=disc_criterion,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            dataloader=dataloader,
            logger=logger,
            config=config,
        )

