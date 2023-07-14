import os
import re
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm

from generate_batch import generate
from scripts.prepare_motion import generate_prompt
from lit_llama.lora import lora_state_dict
from utils.losses import ReConsLoss
from utils.motion_process import recover_from_ric
from visualization.plot_3d_global import plot_3d_motion

def truncate_output_to_eos(output, eos_id):
    try:
        eos_pos = output.tolist().index(eos_id)
        output = output[:eos_pos+1]
    except ValueError:
        pass
    return output


def pad_left(x, max_len, pad_id):
    # pad right based on the longest sequence
    n = max_len - len(x)
    return torch.cat((torch.full((n,), pad_id, dtype=x.dtype).to(x.device), x))


@torch.no_grad()
def plot(tokens, net, dataname):
    meta_dir = "./checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta" if dataname == 't2m' else "./checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta"
    mean = torch.FloatTensor(np.load(os.path.join(meta_dir, 'mean.npy'))).cuda()
    std = torch.FloatTensor(np.load(os.path.join(meta_dir, 'std.npy'))).cuda()

    pred_pose = net.forward_decoder(tokens)
    pred_pose_denorm = pred_pose * std + mean
    pred_xyz = recover_from_ric(pred_pose_denorm, joints_num=22 if dataname == 't2m' else 21).detach().cpu().numpy()[0]
    img = plot_3d_motion([pred_xyz, None, None])
    return pred_xyz, img


@torch.no_grad()        
def vqvae_evaluation(out_dir, val_loader, net, logger, writer, eval_wrapper, nb_iter, best_fid=1000, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, is_train=False): 
    net.eval()
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())


            pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    writer.add_scalar('./Test/FID', fid, nb_iter)
    writer.add_scalar('./Test/Diversity', diversity, nb_iter)
    writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
    writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
    writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
    writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)
    
    if is_train:
        if fid < best_fid : 
            msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
            logger.info(msg)
            best_fid = fid
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

        if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
            msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
            logger.info(msg)
            best_div = diversity
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

        if R_precision[0] > best_top1 : 
            msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
            logger.info(msg)
            best_top1 = R_precision[0]
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

        if R_precision[1] > best_top2 : 
            msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
            logger.info(msg)
            best_top2 = R_precision[1]
        
        if R_precision[2] > best_top3 : 
            msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
            logger.info(msg)
            best_top3 = R_precision[2]
        
        if matching_score_pred < best_matching : 
            msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
            logger.info(msg)
            best_matching = matching_score_pred
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation(val_loader, net, model, logger, tokenizer, eval_wrapper, instruction):
    model.eval()

    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for batch in tqdm(val_loader):
        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

        bs, seq = pose.shape[:2]
        pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()

        encoded = []
        for k in range(bs):
            sample = {"instruction": instruction, "input": clip_text[k]}
            prompt = generate_prompt(sample)
            encoded.append(tokenizer.encode(prompt, bos=True, eos=False, device=model.device).type(torch.int64))

        max_len = max(len(s) for s in encoded)
        encoded = torch.stack([pad_left(x, max_len, pad_id=0) for x in encoded])

        outputs = generate(
            model,
            idx=encoded,
            max_seq_length=seq,
            max_new_tokens=seq,
            temperature=0.8,
            top_k=200,
            eos_id=tokenizer.eos_id
        )

        for k in range(bs):
            output = truncate_output_to_eos(outputs[k].cpu(), tokenizer.eos_id)
            output = tokenizer.decode(output)
            output = output.split("### Response:")[1].strip()
            try:
                output = re.findall(r'\d+', output)
                for j, num in enumerate(output):
                    if int(num) > 511:
                        output = output[:j]
                        break
                if len(output) == 0:
                    index_motion = torch.ones(1, seq).cuda().long()
                else:
                    index_motion = torch.tensor([[int(num) for num in output]]).cuda().long()
            except:
                index_motion = torch.ones(1, seq).cuda().long()

            pred_pose = net.forward_decoder(index_motion)
            cur_len = pred_pose.shape[1]

            pred_len[k] = min(cur_len, seq)
            pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

        pose = pose.cuda().float()

        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
        motion_annotation_list.append(em)
        motion_pred_list.append(em_pred)

        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)

    model.train()
    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, logger


def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists


def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(activations):
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist
