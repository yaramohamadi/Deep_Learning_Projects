import torch
import torch.nn.functional as F

# sprobowac uproszczona wersje lossu z BiGAN bez oddzielnych dyskryminatorow dla X i Z.

class BiGANLoss(torch.nn.Module):
    pass
    # def forward(self, output):
    #     real_loss = self.aggregate_scores(
    #         output["img_real_score"],
    #         output["z_img_score"],
    #         output["comb_real_score"],
    #         generated=False,
    #     )
    #
    #     gen_loss = self.aggregate_scores(
    #         output["img_gen_score"],
    #         output["z_noise_score"],
    #         output["comb_gen_score"],
    #         generated=True,
    #     )
    #     return real_loss + gen_loss


class WGeneratorEncoderLoss(BiGANLoss):
    def forward(self, output):
        comb_output_fake = output["comb_gen_score"]
        comb_output_real = output["comb_real_score"]

        z_output_fake = output["z_noise_score"]
        z_output_real = output["z_img_score"]

        img_output_fake = output["img_gen_score"]
        img_output_real = output["img_real_score"]

        gen_loss = torch.mean(img_output_fake) + torch.mean(comb_output_fake) + torch.mean(z_output_fake)
        real_loss = torch.mean(img_output_real) + torch.mean(comb_output_real) + torch.mean(z_output_real)
        total_loss = - gen_loss + real_loss
        return total_loss / 3


class BiWDiscriminatorLoss(BiGANLoss):
    def forward(self, output):
        comb_output_fake = output["comb_gen_score"]
        comb_output_real = output["comb_real_score"]

        z_output_fake = output["z_noise_score"]
        z_output_real = output["z_img_score"]

        img_output_fake = output["img_gen_score"]
        img_output_real = output["img_real_score"]

        gen_loss = torch.mean(img_output_fake) + torch.mean(comb_output_fake) + torch.mean(z_output_fake)
        real_loss = torch.mean(img_output_real) + torch.mean(comb_output_real) + torch.mean(z_output_real)
        total_loss = - real_loss + gen_loss
        return total_loss / 3


class BiDiscriminatorLoss(BiGANLoss):
    # Loss modes: Add loss_mode ########
    def __init__(self, loss_mode):
        super().__init__()
        self.bce = torch.nn.BCELoss()
        self.loss_mode = loss_mode
    ###################################

    # InfoGAN: Get original C as input ###############################
    def forward(self, output, c_real, c_fake):
   	##################################################################

        true_label = torch.ones_like(output["comb_real_score"])
        false_label = torch.zeros_like(output["comb_real_score"])

        comb_output_fake = output["comb_gen_score"]
        comb_output_real = output["comb_real_score"]

        z_output_fake = output["z_noise_score"]
        z_output_real = output["z_img_score"]

        img_output_fake = output["img_gen_score"]
        img_output_real = output["img_real_score"]
        
        # InfoGAN: predicted Cs ###############
        c_real_predict = output["c_real_predict"]
        c_gen_predict = output["c_gen_predict"]
        #######################################

        # loss modes: ########################################################################
        if self.loss_mode == "all" or self.loss_mode == "info_gan":
          real_output =  torch.mean(F.relu(1. - comb_output_real) + F.relu(1. - z_output_real) + F.relu(1. - img_output_real))
          fake_output =  torch.mean(F.relu(1. + comb_output_fake) + F.relu(1. + z_output_fake) + F.relu(1. + img_output_fake))
          correct_disc = (real_output + fake_output) / 3

        elif self.loss_mode == "no_sx":
          real_output =  torch.mean(F.relu(1. - comb_output_real) + F.relu(1. - z_output_real))
          fake_output =  torch.mean(F.relu(1. + comb_output_fake) + F.relu(1. + z_output_fake))
          correct_disc = (real_output + fake_output) / 2

        elif self.loss_mode == "no_sz":
          real_output =  torch.mean(F.relu(1. - comb_output_real) + F.relu(1. - img_output_real))
          fake_output =  torch.mean(F.relu(1. + comb_output_fake) + F.relu(1. + img_output_fake))
          correct_disc = (real_output + fake_output) / 2

        elif self.loss_mode == "no_sxz":
          real_output =  torch.mean(F.relu(1. - z_output_real) + F.relu(1. - img_output_real))
          fake_output =  torch.mean(F.relu(1. + z_output_fake) + F.relu(1. + img_output_fake))
          correct_disc = (real_output + fake_output) 

        ######################################################################################
        # InfoGAN: first part of loss is calculated above in "all" loss mode. ################
        # now we calculate the second part: information matching 
        if self.loss_mode == "info_gan":
          # maximize estimation of mutual information for both real and fake C
          lambda_ = 1

          for c, c_predicted in [(c_real, c_real_predict), (c_fake, c_gen_predict)]:
            crossentropy_p_q = torch.mean(-torch.sum(c * torch.log(c_predicted + 1e-8), 1))
            # Entropy of p is constant so it can be left out
            I = crossentropy_p_q 
            correct_disc += correct_disc + lambda_ * I
          ######################################################################################

        return correct_disc


# Loss_modes: Add loss modes ############
class GeneratorEncoderLoss(BiGANLoss):
    def __init__(self, loss_mode):
        super().__init__()
        self.bce = torch.nn.BCELoss()
        self.loss_mode = loss_mode
#########################################

    # InfoGAN: Get original C as input ###############################
    def forward(self, output, c_real, c_fake):
   	##################################################################

        true_label = torch.ones_like(output["comb_real_score"])
        false_label = torch.zeros_like(output["comb_real_score"])

        comb_output_fake = output["comb_gen_score"]
        comb_output_real = output["comb_real_score"]

        z_output_fake = output["z_noise_score"]
        z_output_real = output["z_img_score"]

        img_output_fake = output["img_gen_score"]
        img_output_real = output["img_real_score"]

        # InfoGAN: predicted Cs ###############
        c_real_predict = output["c_real_predict"]
        c_gen_predict = output["c_gen_predict"]
        #######################################

        # loss modes: ###############################################
        if self.loss_mode == "all" or self.loss_mode == 'info_gan':
          real_output = torch.mean(img_output_real + z_output_real + comb_output_real)
          fake_output = torch.mean(img_output_fake + z_output_fake + comb_output_fake)
          correct_gen = (real_output - fake_output) / 3

        elif self.loss_mode == "no_sx":
          real_output = torch.mean( z_output_real + comb_output_real)
          fake_output = torch.mean( z_output_fake + comb_output_fake)
          correct_gen = (real_output - fake_output) / 2

        elif self.loss_mode == "no_sz":
          real_output = torch.mean(img_output_real + comb_output_real)
          fake_output = torch.mean(img_output_fake + comb_output_fake)
          correct_gen = (real_output - fake_output) / 2

        elif self.loss_mode == "no_sxz":
          real_output = torch.mean(img_output_real + z_output_real )
          fake_output = torch.mean(img_output_fake + z_output_fake )
          correct_gen = (real_output - fake_output) 
        #############################################################

        # InfoGAN: first part of loss is calculated above in "all" loss mode. ###############
        # now we calculate the second part: information matching 

        #print("_________________________")
        #print(c_real, c_real_predict)
        #print("haaaa")
        #print(c_fake, c_gen_predict)

        if self.loss_mode == "info_gan":
          # maximize estimation of mutual information for both real and fake C
          lambda_ = 1
          for c, c_predicted in [(c_real, c_real_predict), (c_fake, c_gen_predict)]:
            crossentropy_p_q = torch.mean(-torch.sum(c * torch.log(c_predicted + 1e-8), 1))
            I = crossentropy_p_q 
            correct_gen += correct_gen + lambda_ * I
          #####################################################################################

        return correct_gen

    # def aggregate_scores(self, *args, generated):
    #     inputs = torch.cat(args, dim=-1)
    #     summed_inputs = torch.sum(inputs, dim=-1)
    #     if generated: summed_inputs = summed_inputs * -1
    #     loss = torch.mean(summed_inputs)
    #     return loss
