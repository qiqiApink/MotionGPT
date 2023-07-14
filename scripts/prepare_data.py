import os
import sys
from pathlib import Path
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import torch
import numpy as np
import models.vqvae as vqvae
from dataloader.tokenizer_loader import DATALoader
from options import option

args = option.get_args_parser()
args.vq_dir= "./dataset/KIT-ML/VQVAE" if args.dataname == 'kit' else "./dataset/HumanML3D/VQVAE"
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

token_loader = DATALoader(args.dataname, 1, unit_length=2**args.down_t)

net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate)

vqvae_pth = f"./checkpoints/pretrained_vqvae/{args.dataname}.pth"
print ('loading checkpoint from {}'.format(vqvae_pth))
ckpt = torch.load(vqvae_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

for batch in token_loader:
    pose, name = batch
    bs = pose.shape[0]

    pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len
    target = net.encode(pose)
    target = target.cpu().numpy()

    np.save(os.path.join(args.vq_dir, name[0] +'.npy'), target[0])
