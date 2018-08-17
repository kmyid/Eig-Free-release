# test.py ---
#
# Filename: test.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Tue Apr  3 14:17:51 2018 (-0700)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C)
# Visual Computing Group @ University of Victoria
# Computer Vision Lab @ EPFL

# Code:

import datetime
import os
import sys
import time
# from time import time
import IPython

import numpy as np
from parse import parse

import cv2
from six.moves import xrange
from transformations import quaternion_from_matrix
from utils import loadh5, saveh5

def compute_error_for_find_essential(x1, x2, E):
    # x1.shape == x2.shape == (Np, 3)
    Ex1 = E.dot(x1.T)
    Etx2 = E.T.dot(x2.T)

    Ex1 = E.dot(x1.T)
    Etx2 = E.T.dot(x2.T)
    x2tEx1 = np.sum(x2.T * Ex1, axis=0)

    a = Ex1[0] * Ex1[0]
    b = Ex1[1] * Ex1[1]
    c = Etx2[0] * Etx2[0]
    d = Etx2[1] * Etx2[1]

    err = x2tEx1 * x2tEx1 / (a + b + c + d)

    return err


def ourFindEssentialMat(np1, np2, method=cv2.RANSAC, iter_num=1000,
                        threshold=0.01, probs=None, weighted=False,
                        use_prob=True):
    """Python implementation of OpenCV findEssentialMat.

    We have this to try multiple different options for RANSAC, e.g. MLESAC,
    which we simply resorted to doing nothing at the end and not using this
    function.

    """

    min_pt_num = 5
    Np = np1.shape[0]
    perms = np.arange(Np, dtype=np.int)

    best_E = None
    best_inliers = None
    best_err = np.inf

    _np1 = np.concatenate([np1, np.ones((Np, 1))], axis=1)
    _np2 = np.concatenate([np2, np.ones((Np, 1))], axis=1)

    thresh2 = threshold * threshold

    for n in range(iter_num):
        # Randomly select depending on the probability (if given)
        if probs is not None:
            probs /= np.sum(probs)
        if use_prob:
            cur_subs = np.random.choice(
                perms, min_pt_num, replace=False, p=probs)
        else:
            cur_subs = np.random.choice(
                perms, min_pt_num, replace=False, p=None)

        sub_np1 = np1[cur_subs, :]
        sub_np2 = np2[cur_subs, :]
        Es, mask = cv2.findEssentialMat(
            sub_np1, sub_np2, focal=1, pp=(0, 0), method=cv2.RANSAC)
        if Es is None:
            # print('E is None @ {} iteration'.format(n))
            continue

        for i in range(0, Es.shape[0], 3):
            E = Es[i:i + 3, :]
            err = compute_error_for_find_essential(_np1, _np2, E)

            inliers = err <= thresh2
            if method == cv2.RANSAC:
                if weighted:
                    num_inliers = (inliers * probs).sum()
                else:
                    num_inliers = inliers.sum()
                sum_err = -num_inliers
            elif method == 'MLESAC':  # worse than RANSAC
                if weighted:
                    sum_err = (np.abs(err) * probs).sum()
                else:
                    sum_err = np.abs(err).sum()
            if sum_err < best_err:
                best_err = sum_err
                best_E = E
                best_inliers = inliers

    best_inliers = best_inliers.reshape(-1, 1).astype(np.uint8)

    return best_E, best_inliers


def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):
    '''

    :param R_gt: (3,3) or (4,4)
    :param t_gt:
    :param R: (3,3) or (4,4)
    :param t:
    :param q_gt:
    :return:
    '''
    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15
    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t


def eval_nondecompose(p1s, p2s, p3s, T_hat, dR, dt, scores):

    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    th = np.sort(scores)[::-1][num_top]
    mask = scores >= th
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    p3s_good = p3s[mask]

    Rij = dR[:, 0, :].reshape(3, 3)
    Rik = dR[:, 2, :].reshape(3, 3).T
    tij = dt[:, 0, :].reshape(3, 1)
    tki = dt[:, 2, :]
    tik = - np.matmul(Rik, tki.reshape(3, 1))

    # the following code is used for testing.
    # Rij = dR[0, :].reshape(3,3)
    # Rik = dR[1, :].reshape(3,3)
    # tij = dt[0, :]
    # tik = dt[1, :]
    # --------------------------------------

    num_inlier = 0
    # Match types
    T_hat = T_hat.reshape(3, 3, 3).astype(p1s.dtype)
    if p1s_good.shape[0] >= 6:
        # retrieve fundamental matrix from trifocal tensor.
        F21, F31 = fundamentalMatFormTrifocal(T_hat)
        # actually this fundamental matrix is essential matrix.

        num_inlier_21, R_21, t_21, mask_21 = cv2.recoverPose(
            F21, p1s_good, p2s_good)
        num_inlier_31, R_31, t_31, mask_31 = cv2.recoverPose(
            F31, p1s_good, p3s_good)
        mask_new = mask_21 & mask_31
        try:
            err_q_21, err_t_21 = evaluate_R_t(Rij, tij, R_21, t_21)
            err_q_31, err_t_31 = evaluate_R_t(Rik, tik, R_31, t_31)
            err_q = np.maximum(err_q_21, err_q_31)
            err_t = np.maximum(err_t_21, err_t_31)
        except:
            print("Failed in evaluation")
            print(R_21)
            print(t_21)
            print(R_31)
            print(t_31)
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)
    # Change mask type
    mask = mask.flatten().astype(bool)
    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new
        num_inlier = np.sum(mask.flatten().astype(float))
    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated


def eval_decompose(p1s, p2s, p3s, dR, dt, mask=None, method=cv2.LMEDS, probs=None,
                   weighted=False, use_prob=True):
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    # Mask the ones that will not be used
    p1s_good = p1s[mask]
    p2s_good = p2s[mask]
    p3s_good = p3s[mask]

    Rij = dR[:, 0, :].reshape(3, 3)
    Rik = dR[:, 2, :].reshape(3, 3).T
    tij = dt[:, 0, :].reshape(3, 1)
    tki = dt[:, 2, :]
    tik = - np.matmul(Rik, tki.reshape(3, 1))

    # the following code is used for testing.
    # Rij = dR[0, :].reshape(3, 3)
    # Rik = dR[1, :].reshape(3, 3)
    # tij = dt[0, :]
    # tik = dt[1, :]


    probs_good = None
    if probs is not None:
        probs_good = probs[mask]

    num_inlier = 0
    mask_new2 = None
    if p1s_good.shape[0] >= 6:
        if probs is None and method != "MLESAC":
            try:
                T, mask_new = trifocal_from_pts(p1s_good, p2s_good, p3s_good)
            except:
                IPython.embed()
                T = None
        else:
            # E, mask_new = ourFindEssentialMat(
            #     p1s_good, p2s_good, method=method, threshold=0.01,
            #     probs=probs_good, weighted=weighted, use_prob=use_prob)
            raise NotImplementedError("TODO")

        if T is not None:
            F21, F31 = fundamentalMatFormTrifocal(T)
            _, R_21, t_21, mask_21 = cv2.recoverPose(
                F21, p1s_good, p2s_good, mask=mask_new)
            _, R_31, t_31, mask_31 = cv2.recoverPose(
                F31, p1s_good, p3s_good, mask=mask_new)
            # print(R_21)
            # print(t_21)
            # print(R_31)
            # print(t_31)
            mask_new2 = mask_21 & mask_31
            num_inlier = np.sum(mask_new2.flatten().astype(float))
            if num_inlier > 6:
                err_q_21, err_t_21 = evaluate_R_t(Rij, tij, R_21, t_21)
                err_q_31, err_t_31 = evaluate_R_t(Rik, tik, R_31, t_31)
                err_q = np.maximum(err_q_21, err_q_31)
                err_t = np.maximum(err_t_21, err_t_31)
            else:
                err_q = np.pi
                err_t = np.pi / 2
        else:
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new2 is not None:
        # Change mask type
        mask_new2 = mask_new2.flatten().astype(bool)
        mask_updated[mask] = mask_new2
    num_inlier = np.sum(mask_updated.flatten().astype(float))
    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated


def compute_fundamental(x1, x2):
    """    Computes the fundamental matrix from corresponding points
        (x1,x2 3*n arrays) using the 8 point algorithm.
        Each row in the A matrix below is constructed as
        [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1] """

    n = len(x1)
    if len(x2) != n:
        raise ValueError("Number of points don't match.")

    # make homogeneous
    ones = np.ones((n, 1))
    x1 = np.concatenate([x1, ones], axis=1)
    x2 = np.concatenate([x2, ones], axis=1)

    # build matrix for equations
    A = np.matmul(x2.reshape(n, 3, 1), x1.reshape(n, 1, 3)).reshape(n, 9)

    # compute linear least square solution
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # constrain F
    # make rank 2 by zeroing out last singular value
    U, S, V = np.linalg.svd(F)
    S[2] = 0
    F = np.dot(U, np.dot(np.diag(S), V))

    return F / F[2, 2]


def eval_decompose_8points(p1s, p2s, dR, dt, mask=None, method=None):
    if mask is None:
        mask = np.ones((len(p1s),), dtype=bool)
    # Change mask type
    mask = mask.flatten().astype(bool)

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    num_inlier = 0
    mask_new = None
    if p1s_good.shape[0] >= 8:
        E = compute_fundamental(p1s_good, p2s_good)
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E, p1s_good, p2s_good)
        err_q, err_t = evaluate_R_t(dR, dt, R, t)
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new

    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated


def dump_val_res(img1, img2, x1, x2, mask_before, mask_after, cx1, cy1, f1,
                 cx2, cy2, f2, R, t, dump):

    if not os.path.exists(dump):
        os.makedirs(dump)

    # Images
    img1 = img1.transpose(1, 2, 0)
    img2 = img2.transpose(1, 2, 0)
    cv2.imwrite(os.path.join(dump, "img1.png"), img1)
    cv2.imwrite(os.path.join(dump, "img2.png"), img2)

    dump_dict = {}
    dump_dict["x1"] = x1
    dump_dict["cx1"] = cx1
    dump_dict["cy1"] = cy1
    dump_dict["f1"] = f1
    dump_dict["x2"] = x2
    dump_dict["cx2"] = cx2
    dump_dict["cy2"] = cy2
    dump_dict["f2"] = f2
    dump_dict["R"] = R
    dump_dict["t"] = t

    if mask_before is not None:
        dump_dict["mask_before"] = mask_before
    if mask_after is not None:
        dump_dict["mask_after"] = mask_after

    saveh5(dump_dict, os.path.join(dump, "dump.h5"))


def test_process(mode, sess,
                 cur_global_step, merged_summary_op, summary_writer,
                 x, y, R, t, is_training,
                 img1, img2, r,
                 logits_mean, T_hat, loss,
                 data,
                 res_dir, config, va_res_only=False):

    import tensorflow as tf

    time_us = []
    time_ransac_us = []
    time_ransac = []

    inlier_us = []
    inlier_ransac = []
    inlier_ransac_us = []

    if mode == "test":
        print("[{}] {}: Start testing".format(config.data_tr, time.asctime()))

    # Unpack some references
    xs = data["xs"]
    ys = data["ys"]
    Rs = data["Rs"]
    ts = data["ts"]
    # img1s = data["img1s"]
    # cx1s = data["cx1s"]
    # cy1s = data["cy1s"]
    # f1s = data["f1s"]
    # img2s = data["img2s"]
    # cx2s = data["cx2s"]
    # cy2s = data["cy2s"]
    # f2s = data["f2s"]
 
    # Validation
    num_sample = len(xs)

    test_list = [
        "ours",
        # "ours_ransac",
    ]
    if not va_res_only:
        test_list += [
            "ours_ransac",
            # "ours_usac5point",
            # "ours_usac8point",
            # "ours_usacnolo5point",
            # "ours_usacnolo8point",
            # "ours_ransac_weighted",
            # "ours_8point",
            # "ours_top64_ransac",
            # "ours_prob_ransac",
            # "ours_prob_ransac_weighted",
            # "ours_prob_mlesac",
            # "ours_prob_mlesac_weighted",
        ]

    eval_res = {}
    measure_list = ["err_q", "err_t", "num"]
    for measure in measure_list:
        eval_res[measure] = {}
        for _test in test_list:
            eval_res[measure][_test] = np.zeros(num_sample)

    T_hats = []
    y_hats = []
    # Run every test independently. might have different number of keypoints
    for idx_cur in xrange(num_sample):
        # Use minimum kp in batch to construct the batch
        _xs = np.array(
            xs[idx_cur][:, :, :]
        ).reshape(1, 1, -1, 8)
        _ys = np.array(
            ys[idx_cur][:]
        ).reshape(1, -1, 1)
        _dR = np.array(Rs[idx_cur]).reshape(1, 3, 9)
        _dt = np.array(ts[idx_cur]).reshape(1, 3, 3)
        # Create random permutation indices
        feed_dict = {
            x: _xs,
            y: _ys,
            R: _dR,
            t: _dt,
            is_training: config.net_bn_test_is_training,
        }
        fetch = {
            "T_hat": T_hat,  # trifocal tensor by using 6 points algorithm.
            "y_hat": logits_mean,  # label predicted by the network.
            "loss": loss,
            # "summary": merged_summary_op,
            # "global_step": global_step,
        }

        # print("Running network for {} correspondences".format(
        #     _xs.shape[2]
        # ))
        time_start = datetime.datetime.now()
        res = sess.run(fetch, feed_dict=feed_dict)
        time_end = datetime.datetime.now()
        time_diff = time_end - time_start
        # print("Runtime in milliseconds: {}".format(
        #     float(time_diff.total_seconds() * 1000.0)
        # ))
        time_us += [time_diff.total_seconds() * 1000.0]
        T_hats.append(res["T_hat"])
        y_hats.append(res["y_hat"])

    for cur_val_idx in xrange(num_sample):
        _xs = xs[cur_val_idx][:, :, :].reshape(1, 1, -1, 8)
        # _ys = ys[cur_val_idx][:, :].reshape(1, -1, 1)
        _dR = Rs[cur_val_idx].reshape(1, 3, 9)
        _dt = ts[cur_val_idx].reshape(1, 3, 3)  # todo: these data are used for what?

        T_hat_out = T_hats[cur_val_idx].flatten()
        y_hat_out = y_hats[cur_val_idx].flatten()
        if len(y_hat_out) != _xs.shape[2]:
            y_hat_out = np.ones(_xs.shape[2])
        # Eval decompose for all pairs
        _xs = _xs.reshape(-1, 8)
        # x coordinates
        _x1 = _xs[:, :2]  # (kps, 2)
        _x2 = _xs[:, 2:4]
        _x3 = _xs[:, 6:]  # for the third view
        # current validity from network
        _valid = y_hat_out.flatten()
        # choose top ones (get validity threshold)
        _valid_th = np.sort(_valid)[::-1][config.obj_top_k]
        _relu_tanh = np.maximum(0, np.tanh(_valid))

        # For every things to test
        _use_prob = True
        for _test in test_list:
            if _test == "ours":
                _eval_func = "non-decompose"
                _mask_before = _valid >= max(0, _valid_th)
                _method = None
                _probs = None
                _weighted = False
            elif _test == "ours_ransac":
                _eval_func = "decompose"
                _mask_before = _valid >= max(0, _valid_th)
                _method = cv2.RANSAC
                _probs = None
                _weighted = False

            if _eval_func == "non-decompose":
                # this is just using DLT to get the trifocal tensor.
                _err_q, _err_t, _, _, _num_inlier, \
                    _ = eval_nondecompose(
                        _x1, _x2, _x3, T_hat_out, _dR, _dt, y_hat_out)
                _mask_after = _mask_before

            elif _eval_func == "decompose":
                # "6 points algorithm + RANSAC loop with ours"
                time_start = datetime.datetime.now()
                _err_q, _err_t, _, _, _num_inlier, \
                    _mask_after = eval_decompose(
                        _x1, _x2, _x3, _dR, _dt, mask=_mask_before,
                        method=_method, probs=_probs,
                        weighted=_weighted, use_prob=_use_prob)
                time_end = datetime.datetime.now()
                time_diff = time_end - time_start

                # print("Runtime in milliseconds: {}".format(
                #     float(time_diff.total_seconds() * 1000.0)
                # ))
                # print("RANSAC loop without ours")
                inlier_us += [np.sum(_mask_before)]
                inlier_ransac_us += [np.sum(_mask_after)]
                time_ransac_us += [time_diff.total_seconds() * 1000.0]
                time_start = datetime.datetime.now()
                _, _, _, _, _, \
                    _mask_tmp = eval_decompose(
                        _x1, _x2, _x3, _dR, _dt,
                        mask=np.ones_like(_mask_before).astype(bool),
                        method=_method, probs=_probs,
                        weighted=_weighted, use_prob=_use_prob)
                time_end = datetime.datetime.now()
                time_diff = time_end - time_start
                inlier_ransac += [np.sum(_mask_tmp)]
                # print("Runtime in milliseconds: {}".format(
                #     float(time_diff.total_seconds() * 1000.0)
                # ))
                time_ransac += [time_diff.total_seconds() * 1000.0]

            # Load them in list
            eval_res["err_q"][_test][cur_val_idx] = _err_q
            eval_res["err_t"][_test][cur_val_idx] = _err_t
            eval_res["num"][_test][cur_val_idx] = _num_inlier

            if config.vis_dump:
                dump_val_res(
                    img1s[cur_val_idx],
                    img2s[cur_val_idx],
                    _x1, _x2, _mask_before, _mask_after,
                    cx1s[cur_val_idx],
                    cy1s[cur_val_idx],
                    f1s[cur_val_idx],
                    cx2s[cur_val_idx],
                    cy2s[cur_val_idx],
                    f2s[cur_val_idx],
                    Rs[cur_val_idx],
                    ts[cur_val_idx],
                    os.path.join(
                        res_dir, mode, "match", _test,
                        "pair{:08d}".format(cur_val_idx)
                    ),
                )

        # print("Test {}".format(_test))
        # print("Time taken to compute us {}".format(np.median(time_us)))
        # print("Time taken to compute ransac {}".format(np.median(time_ransac)))
        # print("Time taken to compute ransac after us {}".format(
        #     np.median(time_ransac_us)))
        # print("Inliers us {}".format(np.median(inlier_us)))
        # print("Inliers ransac {}".format(np.median(inlier_ransac)))
        # print("Inliers ransac + us {}".format(np.median(inlier_ransac_us)))

    if config.vis_dump:
        print("[{}] {}: End dumping".format(
            config.data_tr, time.asctime()))
        assert config.run_mode != "train"
        return np.nan

    summaries = []
    ret_val = 0
    for _tag in test_list:
        for _sub_tag in measure_list:
            summaries.append(
                tf.Summary.Value(
                    tag="ErrorComputation/" + _tag,
                    simple_value=np.median(eval_res[_sub_tag][_tag])
                )
            )

            # For median error
            ofn = os.path.join(
                res_dir, mode, "median_{}_{}.txt".format(_sub_tag, _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(
                    np.median(eval_res[_sub_tag][_tag])))

        ths = np.arange(7) * 5
        cur_err_q = np.array(eval_res["err_q"][_tag]) * 180.0 / np.pi
        cur_err_t = np.array(eval_res["err_t"][_tag]) * 180.0 / np.pi
        # Get histogram
        q_acc_hist, _ = np.histogram(cur_err_q, ths)
        t_acc_hist, _ = np.histogram(cur_err_t, ths)
        qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
        num_pair = float(len(cur_err_q))
        q_acc_hist = q_acc_hist.astype(float) / num_pair
        t_acc_hist = t_acc_hist.astype(float) / num_pair
        qt_acc_hist = qt_acc_hist.astype(float) / num_pair
        q_acc = np.cumsum(q_acc_hist)
        t_acc = np.cumsum(t_acc_hist)
        qt_acc = np.cumsum(qt_acc_hist)
        # Store return val
        if _tag == "ours":
            ret_val = np.mean(qt_acc[:4])  # 1 == 5
        for _idx_th in xrange(1, len(ths)):
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_q_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(q_acc[:_idx_th]),
                )
            ]
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_t_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(t_acc[:_idx_th]),
                )
            ]
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_qt_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(qt_acc[:_idx_th]),
                )
            ]
            # for q_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_q_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_t_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_qt_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    summary_writer.add_summary(
        tf.Summary(value=summaries), global_step=cur_global_step)

    if mode == "test":
        print("[{}] {}: End testing".format(
            config.data_tr, time.asctime()))

    # Return qt_auc20 of ours
    return ret_val


def comp_process(mode, data, res_dir, config):

    # Unpack some references
    xs = data["xs"]
    ys = data["ys"]
    Rs = data["Rs"]
    ts = data["ts"]
    img1s = data["img1s"]
    cx1s = data["cx1s"]
    cy1s = data["cy1s"]
    f1s = data["f1s"]
    img2s = data["img2s"]
    cx2s = data["cx2s"]
    cy2s = data["cy2s"]
    f2s = data["f2s"]

    # Make fs numpy array
    f1s = np.array(f1s)
    f2s = np.array(f2s)

    # Prepare directory
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print("[{}] {}: Start testing".format(
        config.data_tr, time.asctime()))
    # Validation
    num_sample = len(xs)

    if config.use_lift:
        comp_list = [
            "lmeds", "ransac", "mlesac",
            # "usac5point", "usac8point",
            # "usacnolo5point", "usacnolo8point"
        ]
    else:
        # comp_list = [
        #     "lmeds", "ransac", "top8_8point", "top50_lmeds", "top50_ransac",
        #     "gms", "gms_orb", "gms_default",
        #     "gms_orb_resize", "gms_orb_resize_ransac",
        #     "gms_orb_resize_tt", "gms_orb_resize_ransac_tt",
        #     "mlesac",
        # ]
        comp_list = [
            "lmeds", "ransac", "mlesac",
            # "usac5point", "usac8point",
            # "usacnolo5point", "usacnolo8point"
        ]
        if config.obj_num_kp == 2000:
            comp_list += [
                "gms_orb_resize_ransac_tt",
                "gms_orb_resize_tt",
            ]

    # Initialize arrays that will store measurements
    err_q = {}
    err_t = {}
    num = {}
    for _comp in comp_list:
        err_q[_comp] = np.zeros(num_sample)
        err_t[_comp] = np.zeros(num_sample)
        num[_comp] = np.zeros(num_sample)

    NUM_KP = config.obj_num_kp
    # batch_size = config.val_batch_size
    # num_batch = int(len(xs) / batch_size)

    from gms_matcher import GmsMatcher

    # SIFT
    sift = cv2.xfeatures2d.SIFT_create(
        nfeatures=NUM_KP, contrastThreshold=1e-5)
    if cv2.__version__.startswith('3'):
        sift_matcher = cv2.BFMatcher(cv2.NORM_L2)
    else:
        sift_matcher = cv2.BFMatcher_create(cv2.NORM_L2)
    sift_gms = GmsMatcher(sift, sift_matcher)

    # ORB
    orb = cv2.ORB_create(10000)
    orb.setFastThreshold(0)
    if cv2.__version__.startswith('3'):
        orb_matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        orb_matcher = cv2.BFMatcher_create(cv2.NORM_HAMMING)
    orb_gms = GmsMatcher(orb, orb_matcher)

    for method_name in comp_list:

        # Check res_dir if we have the dump ready
        dump_dir = os.path.join(res_dir, mode, method_name)
        if not os.path.exists(dump_dir):
            os.makedirs(dump_dir)
        # Check for full dump
        full_dump_file = os.path.join(dump_dir, "qtn_all.h5")
        if not os.path.exists(full_dump_file) or config.vis_dump:
            for _idx in xrange(num_sample):
                print("\rWorking on {} / {}".format(
                    _idx + 1, num_sample), end="")
                sys.stdout.flush()
                dump_file = os.path.join(dump_dir, "qtn_{}.txt".format(_idx))
                # If dump exists, just load it
                if os.path.exists(dump_file) and not config.vis_dump:
                    with open(dump_file, "r") as ifp:
                        dump_res = ifp.read()
                    dump_res = parse(
                        "{err_q:e}, {err_t:e}, {num_inlier:d}\n", dump_res)
                    _err_q = dump_res["err_q"]
                    _err_t = dump_res["err_t"]
                    _num_inlier = dump_res["num_inlier"]
                else:
                    # Otherwise compute
                    _xs = xs[_idx][:, :, :].reshape(1, 1, -1, 4)
                    _ys = ys[_idx][:, :].reshape(1, -1, 2)
                    _dR = Rs[_idx]
                    _dt = ts[_idx]
                    # Eval decompose for all pairs
                    _xs = _xs.reshape(-1, 4)
                    # x coordinates
                    _x1 = _xs[:, :2]
                    _x2 = _xs[:, 2:]

                    # Prepare input
                    if method_name == "lmeds":
                        eval_func = eval_decompose
                        _method = cv2.LMEDS
                        _mask = None
                    elif method_name == "ransac":
                        eval_func = eval_decompose
                        _method = cv2.RANSAC
                        _mask = None
                    elif method_name == "mlesac":
                        eval_func = eval_decompose
                        _method = "MLESAC"
                        _mask = None
                    elif method_name == "gms":
                        eval_func = eval_decompose_8points
                        _method = None
                        sift_gms.empty_matches()
                        _x1, _x2, _mask = sift_gms.compute_matches(
                            np.transpose(img1s[_idx], (1, 2, 0)),
                            np.transpose(img2s[_idx], (1, 2, 0)),
                            cx1s[_idx], cx2s[_idx],
                            cy1s[_idx], cy2s[_idx],
                            f1s[_idx], f2s[_idx],
                            with_scale=True, with_rotation=True
                        )
                    elif method_name == "gms_default":
                        eval_func = eval_decompose_8points
                        _method = None
                        orb_gms.empty_matches()
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            np.transpose(img1s[_idx], (1, 2, 0)),
                            np.transpose(img2s[_idx], (1, 2, 0)),
                            cx1s[_idx], cx2s[_idx],
                            cy1s[_idx], cy2s[_idx],
                            f1s[_idx], f2s[_idx],
                            with_scale=False, with_rotation=False
                        )
                    elif method_name == "gms_orb_resize_ransac_tt":
                        eval_func = eval_decompose
                        _method = cv2.RANSAC
                        orb_gms.empty_matches()
                        _img1 = np.transpose(img1s[_idx], (1, 2, 0))
                        _img2 = np.transpose(img2s[_idx], (1, 2, 0))
                        _h1, _w1 = _img1.shape[:2]
                        _h2, _w2 = _img2.shape[:2]
                        _s1 = 480.0 / _h1
                        _s2 = 480.0 / _h2
                        _h1 = int(_h1 * _s1)
                        _w1 = int(np.round(_w1 * _s1))
                        _h2 = int(_h2 * _s2)
                        _w2 = int(np.round(_w2 * _s2))
                        _img1 = cv2.resize(_img1, (_w1, _h1))
                        _img2 = cv2.resize(_img2, (_w2, _h2))
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            _img1, _img2,
                            cx1s[_idx] * _s1, cx2s[_idx] * _s2,
                            cy1s[_idx] * _s1, cy2s[_idx] * _s2,
                            f1s[_idx] * _s1, f2s[_idx] * _s2,
                            with_scale=True, with_rotation=True
                        )
                    elif method_name == "gms_orb_resize_tt":
                        eval_func = eval_decompose_8points
                        _method = None
                        orb_gms.empty_matches()
                        _img1 = np.transpose(img1s[_idx], (1, 2, 0))
                        _img2 = np.transpose(img2s[_idx], (1, 2, 0))
                        _h1, _w1 = _img1.shape[:2]
                        _h2, _w2 = _img2.shape[:2]
                        _s1 = 480.0 / _h1
                        _s2 = 480.0 / _h2
                        _h1 = int(_h1 * _s1)
                        _w1 = int(np.round(_w1 * _s1))
                        _h2 = int(_h2 * _s2)
                        _w2 = int(np.round(_w2 * _s2))
                        _img1 = cv2.resize(_img1, (_w1, _h1))
                        _img2 = cv2.resize(_img2, (_w2, _h2))
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            _img1, _img2,
                            cx1s[_idx] * _s1, cx2s[_idx] * _s2,
                            cy1s[_idx] * _s1, cy2s[_idx] * _s2,
                            f1s[_idx] * _s1, f2s[_idx] * _s2,
                            with_scale=True, with_rotation=True
                        )
                    elif method_name == "gms_orb_resize_ransac":
                        eval_func = eval_decompose
                        _method = cv2.RANSAC
                        orb_gms.empty_matches()
                        _img1 = np.transpose(img1s[_idx], (1, 2, 0))
                        _img2 = np.transpose(img2s[_idx], (1, 2, 0))
                        _h1, _w1 = _img1.shape[:2]
                        _h2, _w2 = _img2.shape[:2]
                        _s1 = 480.0 / _h1
                        _s2 = 480.0 / _h2
                        _h1 = int(_h1 * _s1)
                        _w1 = int(np.round(_w1 * _s1))
                        _h2 = int(_h2 * _s2)
                        _w2 = int(np.round(_w2 * _s2))
                        _img1 = cv2.resize(_img1, (_w1, _h1))
                        _img2 = cv2.resize(_img2, (_w2, _h2))
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            _img1, _img2,
                            cx1s[_idx] * _s1, cx2s[_idx] * _s2,
                            cy1s[_idx] * _s1, cy2s[_idx] * _s2,
                            f1s[_idx] * _s1, f2s[_idx] * _s2,
                            with_scale=False, with_rotation=False
                        )
                    elif method_name == "gms_orb_resize":
                        eval_func = eval_decompose_8points
                        _method = None
                        orb_gms.empty_matches()
                        _img1 = np.transpose(img1s[_idx], (1, 2, 0))
                        _img2 = np.transpose(img2s[_idx], (1, 2, 0))
                        _h1, _w1 = _img1.shape[:2]
                        _h2, _w2 = _img2.shape[:2]
                        _s1 = 480.0 / _h1
                        _s2 = 480.0 / _h2
                        _h1 = int(_h1 * _s1)
                        _w1 = int(np.round(_w1 * _s1))
                        _h2 = int(_h2 * _s2)
                        _w2 = int(np.round(_w2 * _s2))
                        _img1 = cv2.resize(_img1, (_w1, _h1))
                        _img2 = cv2.resize(_img2, (_w2, _h2))
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            _img1, _img2,
                            cx1s[_idx] * _s1, cx2s[_idx] * _s2,
                            cy1s[_idx] * _s1, cy2s[_idx] * _s2,
                            f1s[_idx] * _s1, f2s[_idx] * _s2,
                            with_scale=False, with_rotation=False
                        )
                    elif method_name == "gms_orb":
                        eval_func = eval_decompose_8points
                        _method = None
                        orb_gms.empty_matches()
                        _x1, _x2, _mask = orb_gms.compute_matches(
                            np.transpose(img1s[_idx], (1, 2, 0)),
                            np.transpose(img2s[_idx], (1, 2, 0)),
                            cx1s[_idx], cx2s[_idx],
                            cy1s[_idx], cy2s[_idx],
                            f1s[_idx], f2s[_idx],
                            with_scale=True, with_rotation=True
                        )

                    # Compute errors
                    _err_q, _err_t, _, _, _num_inlier, _mask_after = eval_func(
                        _x1, _x2, _dR, _dt, mask=_mask, method=_method)

                    if config.vis_dump:
                        dump_val_res(
                            img1s[_idx],
                            img2s[_idx],
                            _x1, _x2, _mask, _mask_after,
                            cx1s[_idx],
                            cy1s[_idx],
                            f1s[_idx],
                            cx2s[_idx],
                            cy2s[_idx],
                            f2s[_idx],
                            Rs[_idx],
                            ts[_idx],
                            os.path.join(
                                res_dir, mode, "match", method_name,
                                "pair{:08d}".format(_idx)
                            ),
                        )
                    else:
                        # Write dump
                        with open(dump_file, "w") as ofp:
                            ofp.write("{:e}, {:e}, {:d}\n".format(
                                _err_q, _err_t, _num_inlier))

                # Load them in list
                err_q[method_name][_idx] = _err_q
                err_t[method_name][_idx] = _err_t
                num[method_name][_idx] = _num_inlier

            if not config.vis_dump:
                # Save to full dump
                dump_dict = {}
                dump_dict["err_q"] = err_q[method_name]
                dump_dict["err_t"] = err_t[method_name]
                dump_dict["num"] = num[method_name]
                saveh5(dump_dict, full_dump_file)

            # Remove all intermediate cache
            for _f in os.listdir(dump_dir):
                if _f.startswith("qtn_") and _f.endswith(".txt"):
                    os.remove(os.path.join(dump_dir, _f))

        # Load the full dump file
        else:
            dump_dict = loadh5(full_dump_file)
            err_q[method_name] = dump_dict["err_q"]
            err_t[method_name] = dump_dict["err_t"]
            num[method_name] = dump_dict["num"]

            # Remove all intermediate cache
            for _f in os.listdir(dump_dir):
                if _f.startswith("qtn_") and _f.endswith(".txt"):
                    os.remove(os.path.join(dump_dir, _f))

    print("")

    if config.vis_dump:
        return

    # Report results
    for _tag in comp_list:
        # For median error
        ofn = os.path.join(res_dir, mode, "median_err_q_{}.txt".format(_tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(err_q[_tag])))
        ofn = os.path.join(res_dir, mode, "median_err_t_{}.txt".format(_tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(err_t[_tag])))
        ofn = os.path.join(res_dir, mode, "median_num_{}.txt".format(_tag))
        with open(ofn, "w") as ofp:
            ofp.write("{}\n".format(np.median(num[_tag])))

        # For accuracy AUC
        ths = np.arange(7) * 5
        cur_err_q = np.array(err_q[_tag]) * 180.0 / np.pi
        cur_err_t = np.array(err_t[_tag]) * 180.0 / np.pi
        # Get histogram
        q_acc_hist, _ = np.histogram(cur_err_q, ths)
        t_acc_hist, _ = np.histogram(cur_err_t, ths)
        qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
        num_pair = float(len(cur_err_q))
        q_acc_hist = q_acc_hist.astype(float) / num_pair
        t_acc_hist = t_acc_hist.astype(float) / num_pair
        qt_acc_hist = qt_acc_hist.astype(float) / num_pair
        q_acc = np.cumsum(q_acc_hist)
        t_acc = np.cumsum(t_acc_hist)
        qt_acc = np.cumsum(qt_acc_hist)
        for _idx_th in xrange(1, len(ths)):
            # for q_auc
            ofn = os.path.join(
                res_dir, mode, "acc_q_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
            # for t_auc
            ofn = os.path.join(
                res_dir, mode, "acc_t_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_qt_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    print("[{}] {}: End testing".format(config.data_tr, time.asctime()))


def calcampose(XXc, XXw):
    # kps, 3
    n = np.shape(XXc)[0]
    K = np.eye(n) - np.ones((n, n))/n

    uw = np.mean(XXw, axis=0)
    uc = np.mean(XXc, axis=0)

    sigmx2 = np.square(np.matmul(XXw.T, K))
    sigmx2 = np.mean(np.sum(sigmx2, axis=0))
    SXY = np.matmul(np.matmul(XXc.T, K), XXw)/n
    U, D, V_t = np.linalg.svd(SXY)
    S = np.eye(3)
    #     if np.linalg.det(SXY) < 0:
    #         S[2,2]=-1
    if np.linalg.det(np.matmul(U, V_t)) < 0:
        S[2,2] = -1
    R2 = np.matmul(U, np.matmul(S,V_t))
    D = np.diag(D)
    c2 = np.trace(np.matmul(D,S))/sigmx2
    t2 = uc - c2 * np.matmul(R2, uw)

    X = R2[:, 0]
    Y = R2[:, 1]
    Z = R2[:, 2]
    if np.linalg.norm(np.cross(X, Y) - Z) > 2e-2:
        R2[:, 2] = -Z
    return R2, t2

def DLT(XXw, xx, mask=None):
    x = np.expand_dims(XXw[:, 0], axis=-1)
    y = np.expand_dims(XXw[:, 1], axis=-1)
    z = np.expand_dims(XXw[:, 2], axis=-1)
    u = np.expand_dims(xx[:, 0], axis=-1)
    v = np.expand_dims(xx[:, 1], axis=-1)

    ones = np.ones(np.shape(x))
    zeros = np.zeros(np.shape(x))
    M1n = np.concatenate([x, y, z, ones, zeros, zeros, zeros, zeros, \
                          -u*x, -u*y, -u*z, -u], axis=1)
    M2n = np.concatenate([zeros, zeros, zeros, zeros, x, y, z, ones, \
                          -v*x, -v*y, -v*z, -v], axis=1)
    M = np.concatenate([M1n, M2n], axis=0) # kps, 12
    M_t = np.transpose(M)
    if mask is not None:
        mask = np.ceil(mask)
        if len(np.shape(mask))is not 2:
            mask = np.expand_dims(mask, axis=-1)
        w2n = np.concatenate([mask, mask], axis=0)
        wM = w2n * M
        MwM = np.matmul(M_t, wM)
    else:
        MwM = np.matmul(M_t, M)
    e, V = np.linalg.eigh(MwM)
    v = np.reshape(V[:,0], (3,4))
    v /= np.linalg.norm(v[2,:3])
    v *= np.sign(v[2,3])
    R = v[:,:3]
    t = np.expand_dims(v[:, 3], axis=-1)

    XXc = np.matmul(R, XXw.T) + np.matlib.repmat(t, 1, np.shape(XXw)[0])
    R, t = calcampose(XXc.T, XXw)
    return R,t

def evaluate_R_t_pnp(R_gt, t_gt, R, t, q_gt=None):

    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    err_q = np.linalg.norm(q - q_gt)/(np.linalg.norm(q) + eps)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    err_t = np.linalg.norm(t - t_gt)/(np.linalg.norm(t) + eps)
    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        import IPython
        IPython.embed()

    return err_q, err_t

def test_process(mode, sess, cur_global_step, summary_writer,
                         x, R, t, is_training, data,
                         res_dir, config, va_res_only=False, w=None, delta=None):
    import tensorflow as tf
    # only the pure-DLT would be enough.
    if mode == "test":
        print("[{}] {}: Start testing".format(config.data_tr, time.asctime()))
        xs_mat = []
        ws_mat = []
    # Unpack some references
    xs = data["xs"]
    Rs = data["Rs"]
    ts = data["ts"]
    num_sample = len(xs)

    test_list = [
        "pure_DLT",
    ]
    if va_res_only == True:
        test_list = [
            "pure_DLT",
        ]

    eval_res = {}
    measure_list = ["err_q", "err_t"]

    for measure in measure_list:
        eval_res[measure] = {}
        for _test in test_list:
            eval_res[measure][_test] = np.zeros(num_sample)

    # Run every test independently.
    for idx_cur in xrange(num_sample):
        _xs = np.array(
            xs[idx_cur][:, :, :]
        ).reshape(1, 1, -1, 5)
        _dR = np.array(Rs[idx_cur]).reshape(1, 3, 3)
        _dt = np.array(ts[idx_cur]).reshape(1, 3)
        feed_dict = {
            x: _xs,
            R: _dR,
            t: _dt,
            is_training: config.net_bn_test_is_training,
        }
        fetch = {
            "w": w,
        }
        res = sess.run(fetch, feed_dict=feed_dict)

        if delta is not None:
            xx = _xs[0, 0, :, :2] + res["delta"][0]
        else:
            xx = _xs[0, 0, :, :2]

        XXw = _xs[0, 0, :, 2:]

        if w is not None:
            R_dlt, t_dlt = DLT(XXw, xx, res["w"][0])
        else:
            R_dlt, t_dlt = DLT(XXw, xx, None)

        _err_q, _err_t = evaluate_R_t_pnp(_dR[0], _dt.reshape(3,1), R_dlt, t_dlt)
        eval_res["err_q"]["pure_DLT"][idx_cur] = _err_q
        eval_res["err_t"]["pure_DLT"][idx_cur] = _err_t
        if mode == "test":
            xs_mat.append(np.concatenate([xx, XXw], axis=1))
            ws_mat.append(res["w"][0])

    summaries = []
    ret_val = 0
    for _tag in test_list:
        for _sub_tag in measure_list:
            summaries.append(
                tf.Summary.Value(
                    tag="ErrorComputation/" + _tag,
                    simple_value=np.median(eval_res[_sub_tag][_tag])
                )
            )

            # For median error
            ofn = os.path.join(
                res_dir, mode, "median_{}_{}.txt".format(_sub_tag, _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(
                    np.median(eval_res[_sub_tag][_tag])))

        ths = np.arange(7) * 0.2
        cur_err_q = np.array(eval_res["err_q"][_tag]) * 180.0 / np.pi
        cur_err_t = np.array(eval_res["err_t"][_tag]) * 180.0 / np.pi
        # Get histogram
        q_acc_hist, _ = np.histogram(cur_err_q, ths)
        t_acc_hist, _ = np.histogram(cur_err_t, ths)
        qt_acc_hist, _ = np.histogram(np.maximum(cur_err_q, cur_err_t), ths)
        num_pair = float(len(cur_err_q))
        q_acc_hist = q_acc_hist.astype(float) / num_pair
        t_acc_hist = t_acc_hist.astype(float) / num_pair
        qt_acc_hist = qt_acc_hist.astype(float) / num_pair
        q_acc = np.cumsum(q_acc_hist)
        t_acc = np.cumsum(t_acc_hist)
        qt_acc = np.cumsum(qt_acc_hist)
        # Store return val
        if _tag == "pure_DLT":
            ret_val = np.mean(qt_acc[:4])  # 1 == 5

        for _idx_th in xrange(1, len(ths)):
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_q_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(q_acc[:_idx_th]),
                )
            ]
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_t_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(t_acc[:_idx_th]),
                )
            ]
            summaries += [
                tf.Summary.Value(
                    tag="ErrorComputation/acc_qt_auc{}_{}".format(
                        ths[_idx_th], _tag),
                    simple_value=np.mean(qt_acc[:_idx_th]),
                )
            ]
            # for q_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_q_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(q_acc[:_idx_th])))
            # for t_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_t_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(t_acc[:_idx_th])))
            # for qt_auc
            ofn = os.path.join(
                res_dir, mode,
                "acc_qt_auc{}_{}.txt".format(ths[_idx_th], _tag))
            with open(ofn, "w") as ofp:
                ofp.write("{}\n".format(np.mean(qt_acc[:_idx_th])))

    summary_writer.add_summary(
        tf.Summary(value=summaries), global_step=cur_global_step)
    if mode == "test":
        print("[{}] {}: End testing".format(
            config.data_tr, time.asctime()))
        import scipy.io
        dict_mat = {
            "ws": ws_mat,
            "xs_te": xs,
            "Rs_te": Rs,
            "ts_te": ts,
        }
        scipy.io.savemat("./res/denoise.mat", mdict=dict_mat)
    # Return qt_auc20
    return ret_val

#
# test.py ends here
