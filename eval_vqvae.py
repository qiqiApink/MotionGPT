# This code is based on https://github.com/Mael-zys/T2M-GPT.git
import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vqvae as option_vq
import utils.utils_model as utils_model
from dataloader.eval_loader import DATALoader
from utils.evaluate import vqvae_evaluation
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
from utils.word_vectorizer import WordVectorizer
import warnings
warnings.filterwarnings('ignore')
import numpy as np

args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
os.makedirs(args.out_dir, exist_ok = True)

def main():
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    args.nb_joints = 21 if args.dataname == 'kit' else 22

    val_loader = DATALoader(args.dataname, 'test', 32, w_vectorizer, unit_length=2**args.down_t)

    net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                        args.nb_code,
                        args.code_dim,
                        args.output_emb_width,
                        args.down_t,
                        args.stride_t,
                        args.width,
                        args.depth,
                        args.dilation_growth_rate,
                        args.vq_act,
                        args.vq_norm)

    if args.resume_pth : 
        logger.info('loading checkpoint from {}'.format(args.resume_pth))
        ckpt = torch.load(args.resume_pth, map_location='cpu')
        net.load_state_dict(ckpt['net'], strict=True)
    net.train()
    net.cuda()

    fid = []
    div = []
    top1 = []
    top2 = []
    top3 = []
    matching = []
    repeat_time = 20
    for _ in range(repeat_time):
        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = vqvae_evaluation(args.out_dir, val_loader, net, logger, writer, eval_wrapper, 0)
        fid.append(best_fid)
        div.append(best_div)
        top1.append(best_top1)
        top2.append(best_top2)
        top3.append(best_top3)
        matching.append(best_matching)
    print('final result:')
    print('fid: ', sum(fid)/repeat_time)
    print('div: ', sum(div)/repeat_time)
    print('top1: ', sum(top1)/repeat_time)
    print('top2: ', sum(top2)/repeat_time)
    print('top3: ', sum(top3)/repeat_time)
    print('matching: ', sum(matching)/repeat_time)

    fid = np.array(fid)
    div = np.array(div)
    top1 = np.array(top1)
    top2 = np.array(top2)
    top3 = np.array(top3)
    matching = np.array(matching)
    msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}"
    logger.info(msg_final)


if __name__ == '__main__':
    main()