hparams = {
    # training utils
    "seed": 420,
    "device": "cuda",
    "img_rows": 4,
    "save_img_count": 12,
    "real_imgs_save_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/{ds_name}/{model_architecture}/{loss_mode}/real_img/{hparams}",
    "gen_imgs_save_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/{ds_name}/{model_architecture}/{loss_mode}/gen_img/{hparams}",
    "logging_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/{ds_name}/{model_architecture}/{loss_mode}/logs/{name}",
    "save_model_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/{ds_name}/{model_architecture}/{loss_mode}/checkpoints/",
    "rec_imgs_save_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/{ds_name}/{model_architecture}/{loss_mode}/rec_img/{hparams}",
    "save_org_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/org/",
    "save_gen_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/{ds_name}/{model_architecture}/{loss_mode}/gen/",
    "save_gen_c_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/{ds_name}/{model_architecture}/{loss_mode}/gen_c/",
    "save_encoded_path": "/content/drive/MyDrive/BigBiGAN-PyTorch-main//data/{ds_name}/{model_architecture}/{loss_mode}/encoded/",
    "save_name": "gan",
    "save_model_interval": 1,

    # hparams
    "clf_lr": 2e-4,
    "disc_steps": 2,
    "gen_steps": 1,
    "epochs": 20,
    "lr_gen": 2e-4,
    "lr_disc": 2e-4,
    "betas": (0.5, 0.999),

    # model params
    "dropout": 0.2,
    "spectral_norm": True,
    "weight_cutoff": 0.00,
    "add_noise": 0,
}
