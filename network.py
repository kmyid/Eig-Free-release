# network.py ---
#
# Filename: network.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Tue Apr  3 12:27:30 2018 (-0700)
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


import os
import sys

import numpy as np
import tensorflow as tf
from parse import parse
from tqdm import trange

from tests import test_process

class OutliersNetwork(object):
    """Network class """

    def __init__(self, config):

        self.config = config

        # Initialize thenosrflow session
        self._init_tensorflow()
        print("Initial setting is done!")
        # Build the network
        self._build_placeholder()
        self._build_preprocessing()
        self._build_model()
        self._build_loss()
        self._build_optim()
        self._build_eval()
        self._build_summary()
        self._build_writer()

    def _init_tensorflow(self):
        # limit CPU threads with OMP_NUM_THREADS
        num_threads = os.getenv("OMP_NUM_THREADS", "")
        if num_threads != "":
            num_threads = int(num_threads)
            print("limiting tensorflow to {} threads!".format(
                num_threads
            ))
            # Limit
            tfconfig = tf.ConfigProto(
                intra_op_parallelism_threads=num_threads,
                inter_op_parallelism_threads=num_threads,
            )
        else:
            tfconfig = tf.ConfigProto()
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.Session(config=tfconfig)

    def _build_placeholder(self):
        """Build placeholders."""

        # Make tensforflow placeholder
        self.x_in = tf.placeholder(tf.float32, [None, 1, None, 5], name="x_in")
        self.R_in = tf.placeholder(tf.float32, [None, 3, 3], name="R_in")
        self.t_in = tf.placeholder(tf.float32, [None, 3], name="t_in")
        self.is_training = tf.placeholder(tf.bool, (), name="is_training")

        # Global step for optimization
        self.global_step = tf.get_variable(
            "global_step", shape=(),
            initializer=tf.zeros_initializer(),
            dtype=tf.int64,
            trainable=False)

    def _build_preprocessing(self):
        """Build preprocessing related graph."""

        # For now, do nothing
        pass

    def _build_model(self):
        """Build our MLP network."""

        with tf.variable_scope("Matchnet", reuse=tf.AUTO_REUSE):
            # For determining the runtime shape
            x_shp = tf.shape(self.x_in)

            # -------------------- Network archintecture --------------------
            # Import correct build_graph function
            from archs.cvpr2018 import build_graph
            # Build graph
            print("Building Graph")
            self.logits = build_graph(self.x_in, self.is_training, self.config)
            # ---------------------------------------------------------------

            # Turn into weights for each sample
            self.w = tf.nn.relu(tf.tanh(self.logits)) # bs, n ,2

    def _build_loss(self):
        """Build loss"""

        with tf.variable_scope("Loss", reuse=tf.AUTO_REUSE):
            x = self.x_in
            R = self.R_in
            t = self.t_in
            # delta_u = self.delta[:, :, 0]
            # delta_v = self.delta[:, :, 1]
            Xw = tf.expand_dims(x[:,0,:,2], axis=-1)
            Yw = tf.expand_dims(x[:,0,:,3], axis=-1)
            Zw = tf.expand_dims(x[:,0,:,4], axis=-1)
            ones = tf.ones_like(Xw, dtype=tf.float32)
            zeros = tf.zeros_like(Xw, dtype=tf.float32)
            u = x[:,0,:,0][..., None] #+ delta_u[..., None]
            v = x[:,0,:,1][..., None] #+ delta_v[..., None]

            M1n = tf.concat([Xw, Yw, Zw, ones, zeros, zeros, zeros, zeros, -u*Xw, -u*Yw, -u*Zw, -u],axis=2)
            M2n = tf.concat([zeros, zeros, zeros, zeros, Xw, Yw, Zw, ones, -v*Xw, -v*Yw, -v*Zw, -v],axis=2)
            M = tf.concat([M1n, M2n], axis=1) # bs, 2*kps, 12

            w_2n = tf.concat([self.w, self.w], axis=-1)
            M_t = tf.transpose(M, [0, 2, 1])
            wM = tf.expand_dims(w_2n, axis=-1) * M
            MwM = tf.matmul(M_t, wM)
            x_shp = tf.shape(x)

            with tf.variable_scope("GT"):
                e_gt = tf.reshape(tf.concat([R, tf.expand_dims(t,axis=-1)], axis=-1), (-1,12)) # bs, 12
                e_gt /= tf.norm(e_gt, axis=1, keepdims=True)
                e_gt = tf.expand_dims(e_gt, axis=-1)

            with tf.variable_scope("Denoise"):
                e_gt_t = tf.transpose(e_gt, [0, 2, 1])
                d_term = tf.matmul(tf.matmul(e_gt_t, MwM), e_gt)

                e_hat = tf.eye(12, batch_shape=[x_shp[0]], dtype=tf.float32) - tf.matmul(e_gt, e_gt_t)
                e_hat_t = tf.transpose(e_hat, [0,2,1])
                XwX_e_neg = tf.matmul(tf.matmul(e_hat_t, MwM), e_hat)
                r_term = tf.trace(XwX_e_neg)
                beta = self.config.beta
                alpha = self.config.alpha
                self.loss = tf.reduce_mean(d_term + alpha*tf.exp(- beta*r_term))

                # d_term = tf.squeeze(d_term, axis=-1)
                # r_term = tf.norm(self.delta, axis=-1, ord=2)
                # r_term = tf.reduce_mean(r_term, axis=-1, keepdims=True)
                # beta = self.config.beta
                # self.loss = tf.reduce_mean(d_term + beta * r_term)
            tf.summary.scalar("loss", self.loss)

    def _build_optim(self):
        """Build optimizer related ops and vars."""

        with tf.variable_scope("Optimization", reuse=tf.AUTO_REUSE):
            learning_rate = self.config.train_lr
            max_grad_norm = None
            optim = tf.train.AdamOptimizer(learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                grads_and_vars = optim.compute_gradients(self.loss)

                # gradient clipping
                if max_grad_norm is not None:
                    new_grads_and_vars = []
                    for idx, (grad, var) in enumerate(grads_and_vars):
                        if grad is not None:
                            new_grads_and_vars.append((
                                tf.clip_by_norm(grad, max_grad_norm), var))
                    grads_and_vars = new_grads_and_vars

                # Check numerics and report if something is going on. This
                # will make the backward pass stop and skip the batch
                new_grads_and_vars = []
                for idx, (grad, var) in enumerate(grads_and_vars):
                    if grad is not None:
                        grad = tf.check_numerics(
                            grad, "Numerical error in gradient for {}"
                                  "".format(var.name))
                    new_grads_and_vars.append((grad, var))

                # Should only apply grads once they are safe
                self.optim = optim.apply_gradients(
                    new_grads_and_vars, global_step=self.global_step)

            # # Summarize all gradients
            # for grad, var in grads_and_vars:
            #     if grad is not None:
            #         tf.summary.histogram(var.name + '/gradient', grad)

    def _build_eval(self):
        """Build the evaluation related ops"""

        # We use a custom evaluate function. No building here...
        pass

    def _build_summary(self):
        """Build summary ops."""

        # Merge all summary op
        self.summary_op = tf.summary.merge_all()

    def _build_writer(self):
        """Build the writers and savers"""

        # Create suffix automatically if not provided
        suffix_tr = self.config.log_dir
        if suffix_tr == "":
            suffix_tr = "-".join(sys.argv)
        suffix_te = self.config.test_log_dir
        if suffix_te == "":
            suffix_te = suffix_tr

        # Directories for train/test
        self.res_dir_tr = os.path.join(self.config.res_dir, suffix_tr)
        self.res_dir_va = os.path.join(self.config.res_dir, suffix_te)
        self.res_dir_te = os.path.join(self.config.res_dir, suffix_te)

        # Create summary writers
        if self.config.run_mode == "train":
            self.summary_tr = tf.summary.FileWriter(
                os.path.join(self.res_dir_tr, "train", "logs"))
        if self.config.run_mode != "comp":
            self.summary_va = tf.summary.FileWriter(
                os.path.join(self.res_dir_va, "valid", "logs"))
        if self.config.run_mode == "test":
            self.summary_te = tf.summary.FileWriter(
                os.path.join(self.res_dir_te, "test", "logs"))

        # Create savers (one for current, one for best)
        self.saver_cur = tf.train.Saver()
        self.saver_best = tf.train.Saver()
        # Save file for the current model
        self.save_file_cur = os.path.join(
            self.res_dir_tr, "model")
        # Save file for the best model
        self.save_file_best = os.path.join(
            self.res_dir_tr, "models-best")

        # Other savers
        self.va_res_file = os.path.join(self.res_dir_va, "valid", "va_res.txt")

    def train(self, data):
        """Training function.

        Parameters
        ----------
        data_tr : tuple
            Training data.

        data_va : tuple
            Validation data.

        x_va : ndarray
            Validation data.

        y_va : ndarray
            Validation labels.

        """
        # ----------------------------------------
        # Resume data if it already exists
        latest_checkpoint = tf.train.latest_checkpoint(
            self.res_dir_tr)
        b_resume = latest_checkpoint is not None

        if b_resume:
            # Restore network
            print("Restoring from {}...".format(
                self.res_dir_tr))
            self.saver_cur.restore(
                self.sess,
                latest_checkpoint
            )
            # restore number of steps so far
            step = self.sess.run(self.global_step)
            # restore best validation result
            if os.path.exists(self.va_res_file):
                with open(self.va_res_file, "r") as ifp:
                    dump_res = ifp.read()
                dump_res = parse(
                    "{best_va_res:e}\n", dump_res)
                best_va_res = dump_res["best_va_res"]
        else:
            print("Starting from scratch...")
            step = 0
            best_va_res = -1
            print("Initializing all the parameter...")
            self.sess.run(tf.global_variables_initializer())

        # ----------------------------------------
        # Unpack some data for simple coding
        xs_tr = data["train"]["xs"]
        Rs_tr = data["train"]["Rs"]
        ts_tr = data["train"]["ts"]

        # ----------------------------------------
        # The training loop
        batch_size = self.config.train_batch_size
        max_iter = self.config.train_iter
        for step in trange(step, max_iter, ncols=self.config.tqdm_width):

            # ----------------------------------------
            # Batch construction

            # Get a random training batch
            ind_cur = np.random.choice(
                len(xs_tr), batch_size, replace=False)
            # Use minimum kp in batch to construct the batch
            numkps = np.array([xs_tr[_i].shape[1] for _i in ind_cur])
            cur_num_kp = numkps.min()
            # Actual construction of the batch
            xs_b = np.array(
                [xs_tr[_i][:, :cur_num_kp, :] for _i in ind_cur]
            ).reshape(batch_size, 1, cur_num_kp, 5)
            Rs_b = np.array(
                [Rs_tr[_i] for _i in ind_cur]
            ).reshape(batch_size, 3, 3)
            ts_b = np.array(
                [ts_tr[_i] for _i in ind_cur]
            ).reshape(batch_size, 3)

            # ----------------------------------------
            # Train

            # Feed Dict
            feed_dict = {
                self.x_in: xs_b,
                self.R_in: Rs_b,
                self.t_in: ts_b,
                self.is_training: True,
            }
            # Fetch
            fetch = {
                "optim": self.optim,
            }
            # Check if we want to write summary and check validation
            b_write_summary = ((step + 1) % self.config.report_intv) == 0
            b_validate = ((step + 1) % self.config.val_intv) == 0
            if b_write_summary or b_validate:
                fetch["summary"] = self.summary_op
                fetch["global_step"] = self.global_step

            # Run optimization
            try:
                res = self.sess.run(fetch, feed_dict=feed_dict)
            except (ValueError, tf.errors.InvalidArgumentError):
                print("Backward pass had numerical errors. "
                      "This training batch is skipped!")
                continue
            # Write summary and save current model
            if b_write_summary:
                self.summary_tr.add_summary(
                    res["summary"], global_step=res["global_step"])
                self.saver_cur.save(
                    self.sess, self.save_file_cur,
                    global_step=res["global_step"],
                    write_meta_graph=False)

            # ----------------------------------------
            # Validation
            if b_validate:
                va_res = 0
                cur_global_step = res["global_step"]
                va_res = test_process("valid", self.sess, cur_global_step, self.summary_va,
                                              self.x_in, self.R_in, self.t_in, self.is_training,
                                              data["valid"], self.res_dir_va, self.config, True,
                                              w=self.w, delta=None)
                # Higher the better
                if va_res > best_va_res:
                    print(
                        "Saving best model with va_res = {}".format(
                            va_res))
                    best_va_res = va_res
                    # Save best validation result
                    with open(self.va_res_file, "w") as ofp:
                        ofp.write("{:e}\n".format(best_va_res))
                    # Save best model
                    self.saver_best.save(
                        self.sess, self.save_file_best,
                        write_meta_graph=False,
                    )

    def test(self, data):
        """Test routine"""

        # Check if model exists
        if not os.path.exists(self.save_file_best + ".index"):
            print("Model File {} does not exist! Quiting".format(
                self.save_file_best))
            exit(1)

        # Restore model
        print("Restoring from {}...".format(
            self.save_file_best))
        self.saver_best.restore(
            self.sess,
            self.save_file_best)
        # Run Test
        cur_global_step = 0     # dummy

        test_mode_list = ["test"]
        for test_mode in test_mode_list:
            te_res = test_process(test_mode, self.sess, cur_global_step, self.summary_va,
                                          self.x_in, self.R_in, self.t_in, self.is_training,
                                          data[test_mode], self.res_dir_va,
                                          self.config, False, w=self.w, delta=None)

#
# network.py ends here
