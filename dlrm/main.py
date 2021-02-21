import functools
import os
import sys
import time
from typing import Dict

import intel_pytorch_extension as ipex
import numpy as np
import pandas as pd
import ray
import ray.util.data as ml_data
import raydp
import sklearn.metrics
import torch
from intel_pytorch_extension import core
from ray.util.sgd.utils import find_free_port
from torch.utils.data.dataset import IterableDataset

import dlrm.extend_distributed as ext_dist
import dlrm.mlperf_logger as mlperf_logger
from dlrm.dlrm_s_pytorch import DLRM_Net, LRPolicyScheduler
# quotient-remainder trick
# mixed-dimension trick
from dlrm.tricks.md_embedding_bag import md_solver
from .dlrm_s_pytorch import ModelArguments
from .preproc.spark_preproc import pre_process, PreProcArguments


@ray.remote
class Executor:
    def __init__(self,
                 inter_op_parallelism: int,
                 intra_op_parallelism: int,
                 world_size: int,
                 args: ModelArguments):

        torch.set_num_threads(intra_op_parallelism)
        torch.set_num_interop_threads(inter_op_parallelism)

        self.args = args

        self.world_rank = None
        self.world_size = world_size

        self.train_ds = None
        self.test_ds = None

        self.dlrm_net = None
        self.optimizer = None
        self.lr_scheduler = None

        self.loss_ws = None
        self.loss_fn = None

        self.use_gpu = False
        self.use_ipex = False
        self.device = None

    def setup_address(self):
        """This methods will only called when this is the rank zero node"""
        ip = ray.services.get_node_ip_address()
        port = find_free_port()
        return ip, port

    def setup_process_group(self, master_addr, master_port, world_rank):
        self.world_rank = world_rank
        os.environ["RANK"] = str(world_rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port

        ext_dist.init_distributed(world_rank, self.world_size, "gloo")

    def set_dataset(self, train_ds, test_ds):
        fn = functools.partial(collate_fn,
                               num_numerical_features=self.args.arch_dense_feature_size,
                               max_ind_range=self.args.max_ind_range,
                               flag_input_torch_tensor=True)
        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=None,
            batch_sampler=None,
            shuffle=False,
            num_workers=0,
            collate_fn=fn,
            pin_memory=False,
            drop_last=False,
            sampler=None
        )
        self.train_ds = train_loader

        if test_ds is not None:
            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=None,
                batch_sampler=None,
                shuffle=False,
                num_workers=0,
                collate_fn=fn,
                pin_memory=False,
                drop_last=False,
                sampler=None
            )
            self.test_ds = test_loader

    def time_wrap(self, use_gpu):
        if use_gpu:
            torch.cuda.synchronize()
        return time.time()

    def dlrm_wrap(self, X, lS_o, lS_i, use_gpu, use_ipex, device):
        if use_gpu or use_ipex:  # .cuda()
            # lS_i can be either a list of tensors or a stacked tensor.
            # Handle each case below:
            lS_i = [S_i.to(device) for S_i in lS_i] if isinstance(lS_i, list) \
                else lS_i.to(device)
            lS_o = [S_o.to(device) for S_o in lS_o] if isinstance(lS_o, list) \
                else lS_o.to(device)
            return self.dlrm_net(
                X.to(device),
                lS_o,
                lS_i
            )
        else:
            return self.dlrm_net(X, lS_o, lS_i)

    def loss_fn_wrap(self, Z, T, use_gpu, use_ipex, device):
        if self.args.loss_function == "mse" or self.args.loss_function == "bce":
            if use_gpu or use_ipex:
                return self.loss_fn(Z, T.to(device))
            else:
                return self.loss_fn(Z, T)
        elif self.args.loss_function == "wbce":
            if use_gpu:
                loss_ws_ = self.loss_ws[T.data.view(-1).long()].view_as(T).to(device)
                loss_fn_ = self.loss_fn(Z, T.to(device))
            else:
                loss_ws_ = self.loss_ws[T.data.view(-1).long()].view_as(T)
                loss_fn_ = self.loss_fn(Z, T.to(device))
            loss_sc_ = loss_ws_ * loss_fn_
            # debug prints
            # print(loss_ws_)
            # print(loss_fn_)
            return loss_sc_.mean()

    def create_model(self, ln_emb):
        args = self.args
        # some basic setup
        np.random.seed(args.numpy_rand_seed)
        np.set_printoptions(precision=args.print_precision)
        torch.set_printoptions(precision=args.print_precision)
        torch.manual_seed(args.numpy_rand_seed)

        if (args.test_mini_batch_size < 0):
            # if the parameter is not set, use the training batch size
            args.test_mini_batch_size = args.mini_batch_size
        if (args.test_num_workers < 0):
            # if the parameter is not set, use the same parameter for training
            args.test_num_workers = args.num_workers
        if (args.mini_batch_size % ext_dist.my_size !=0 or args.test_mini_batch_size % ext_dist.my_size != 0):
            print("Either test minibatch (%d) or train minibatch (%d) does not split across %d ranks" % (args.test_mini_batch_size, args.mini_batch_size, ext_dist.my_size))
            sys.exit(1)

        use_gpu = args.use_gpu and torch.cuda.is_available()
        self.use_gpu = use_gpu
        use_ipex = args.use_ipex

        if use_gpu:
            torch.cuda.manual_seed_all(args.numpy_rand_seed)
            torch.backends.cudnn.deterministic = True
            if ext_dist.my_size > 1:
                ngpus = torch.cuda.device_count()  # 1
                if ext_dist.my_local_size > torch.cuda.device_count():
                    print("Not sufficient GPUs available... local_size = %d, ngpus = %d" % (ext_dist.my_local_size, ngpus))
                    sys.exit(1)
                ngpus = 1
                device = torch.device("cuda", ext_dist.my_local_rank)
            else:
                device = torch.device("cuda", 0)
                ngpus = torch.cuda.device_count()  # 1
            print("Using {} GPU(s)...".format(ngpus))
        elif use_ipex:
            device = torch.device("dpcpp")
            print("Using IPEX...")
        else:
            device = torch.device("cpu")
            print("Using CPU...")

        self.device = device

        ### prepare training data ###
        ln_bot = np.fromstring(args.arch_mlp_bot, dtype=int, sep="-")
        # input data

        mlperf_logger.barrier()
        mlperf_logger.log_end(key=mlperf_logger.constants.INIT_STOP)
        mlperf_logger.barrier()
        mlperf_logger.log_start(key=mlperf_logger.constants.RUN_START)
        mlperf_logger.barrier()

        # enforce maximum limit on number of vectors per embedding
        if args.max_ind_range > 0:
            ln_emb = np.array(list(map(
                lambda x: min(x, args.max_ind_range),
                ln_emb
            )))
        m_den = args.arch_dense_feature_size
        ln_bot[0] = m_den

        ### parse command line arguments ###
        m_spa = args.arch_sparse_feature_size
        num_fea = ln_emb.size + 1  # num sparse + num dense features
        m_den_out = ln_bot[ln_bot.size - 1]
        if args.arch_interaction_op == "dot":
            # approach 1: all
            # num_int = num_fea * num_fea + m_den_out
            # approach 2: unique
            if args.arch_interaction_itself:
                num_int = (num_fea * (num_fea + 1)) // 2 + m_den_out
            else:
                num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out
        elif args.arch_interaction_op == "cat":
            num_int = num_fea * m_den_out
        else:
            sys.exit(
                "ERROR: --arch-interaction-op="
                + args.arch_interaction_op
                + " is not supported"
            )
        arch_mlp_top_adjusted = str(num_int) + "-" + args.arch_mlp_top
        ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

        # sanity check: feature sizes and mlp dimensions must match
        if m_den != ln_bot[0]:
            sys.exit(
                "ERROR: arch-dense-feature-size "
                + str(m_den)
                + " does not match first dim of bottom mlp "
                + str(ln_bot[0])
            )
        if args.qr_flag:
            if args.qr_operation == "concat" and 2 * m_spa != m_den_out:
                sys.exit(
                    "ERROR: 2 arch-sparse-feature-size "
                    + str(2 * m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                    + " (note that the last dim of bottom mlp must be 2x the embedding dim)"
                )
            if args.qr_operation != "concat" and m_spa != m_den_out:
                sys.exit(
                    "ERROR: arch-sparse-feature-size "
                    + str(m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                )
        else:
            if m_spa != m_den_out:
                sys.exit(
                    "ERROR: arch-sparse-feature-size "
                    + str(m_spa)
                    + " does not match last dim of bottom mlp "
                    + str(m_den_out)
                )
        if num_int != ln_top[0]:
            sys.exit(
                "ERROR: # of feature interactions "
                + str(num_int)
                + " does not match first dimension of top mlp "
                + str(ln_top[0])
            )

        # assign mixed dimensions if applicable
        if args.md_flag:
            m_spa = md_solver(
                torch.tensor(ln_emb),
                args.md_temperature,  # alpha
                d0=m_spa,
                round_dim=args.md_round_dims
            ).tolist()

        # test prints (model arch)
        if args.debug_mode:
            print("model arch:")
            print(
                "mlp top arch "
                + str(ln_top.size - 1)
                + " layers, with input to output dimensions:"
            )
            print(ln_top)
            print("# of interactions")
            print(num_int)
            print(
                "mlp bot arch "
                + str(ln_bot.size - 1)
                + " layers, with input to output dimensions:"
            )
            print(ln_bot)
            print("# of features (sparse and dense)")
            print(num_fea)
            print("dense feature size")
            print(m_den)
            print("sparse feature size")
            print(m_spa)
            print(
                "# of embeddings (= # of sparse features) "
                + str(ln_emb.size)
                + ", with dimensions "
                + str(m_spa)
                + "x:"
            )
            print(ln_emb)

            # print("data (inputs and targets):")
            # for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
            #     # early exit if nbatches was set by the user and has been exceeded
            #     if nbatches > 0 and j >= nbatches:
            #         break
            #
            #     print("mini-batch: %d" % j)
            #     print(X.detach().cpu().numpy())
            #     # transform offsets to lengths when printing
            #     print(
            #         [
            #             np.diff(
            #                 S_o.detach().cpu().tolist() + list(lS_i[i].shape)
            #             ).tolist()
            #             for i, S_o in enumerate(lS_o)
            #         ]
            #     )
            #     print([S_i.detach().cpu().tolist() for S_i in lS_i])
            #     print(T.detach().cpu().numpy())
            #
        ndevices = min(ngpus, args.mini_batch_size, num_fea - 1) if use_gpu else -1

        ### construct the neural network specified above ###
        # WARNING: to obtain exactly the same initialization for
        # the weights we need to start from the same random seed.
        # np.random.seed(args.numpy_rand_seed)
        print('Creating the model...')
        dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            arch_interaction_op=args.arch_interaction_op,
            arch_interaction_itself=args.arch_interaction_itself,
            sigmoid_bot=-1,
            sigmoid_top=ln_top.size - 2,
            sync_dense_params=args.sync_dense_params,
            loss_threshold=args.loss_threshold,
            ndevices=ndevices,
            qr_flag=args.qr_flag,
            qr_operation=args.qr_operation,
            qr_collisions=args.qr_collisions,
            qr_threshold=args.qr_threshold,
            md_flag=args.md_flag,
            md_threshold=args.md_threshold,
            sparse_dense_boundary=args.sparse_dense_boundary,
            bf16 = args.bf16,
            use_ipex = args.use_ipex
        )

        self.dlrm_net = dlrm

        print('Model created!')
        # test prints
        if args.debug_mode:
            print("initial parameters (weights and bias):")
            for param in dlrm.parameters():
                print(param.detach().cpu().numpy())
            # print(dlrm)

        if args.use_ipex:
            dlrm = dlrm.to(device)
            print(dlrm, device, args.use_ipex)

        if use_gpu:
            # Custom Model-Data Parallel
            # the mlps are replicated and use data parallelism, while
            # the embeddings are distributed and use model parallelism
            dlrm = dlrm.to(device)  # .cuda()
            if dlrm.ndevices > 1:
                dlrm.emb_l = dlrm.create_emb(m_spa, ln_emb)

        if ext_dist.my_size > 1:
            if use_gpu:
                device_ids = [ext_dist.my_local_rank]
                dlrm.bot_l = ext_dist.DDP(dlrm.bot_l, device_ids=device_ids)
                dlrm.top_l = ext_dist.DDP(dlrm.top_l, device_ids=device_ids)
            else:
                dlrm.bot_l = ext_dist.DDP(dlrm.bot_l)
                dlrm.top_l = ext_dist.DDP(dlrm.top_l)
                for i in range(len(dlrm.emb_dense)):
                    dlrm.emb_dense[i] = ext_dist.DDP(dlrm.emb_dense[i])

        # specify the loss function
        if args.loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="mean")
        elif args.loss_function == "bce":
            self.loss_fn = torch.nn.BCELoss(reduction="mean")
        elif args.loss_function == "wbce":
            self.loss_ws = torch.tensor(np.fromstring(args.loss_weights, dtype=float, sep="-"))
            self.loss_fn = torch.nn.BCELoss(reduction="none")
        else:
            sys.exit("ERROR: --loss-function=" + args.loss_function + " is not supported")

        if not args.inference_only:
            # specify the optimizer algorithm
            if ext_dist.my_size == 1:
                if args.bf16 and ipex.is_available():
                    optimizer = ipex.SplitSGD(dlrm.parameters(), lr=args.learning_rate)
                else:
                    optimizer = torch.optim.SGD(dlrm.parameters(), lr=args.learning_rate)
            else:
                if args.bf16 and ipex.is_available():
                    optimizer = ipex.SplitSGD([
                        {"params": [p for emb in dlrm.emb_sparse for p in emb.parameters()], "lr" : args.learning_rate / ext_dist.my_size},
                        {"params": [p for emb in dlrm.emb_dense for p in emb.parameters()], "lr" : args.learning_rate},
                        {"params": dlrm.bot_l.parameters(), "lr" : args.learning_rate},
                        {"params": dlrm.top_l.parameters(), "lr" : args.learning_rate}
                    ], lr=args.learning_rate)
                else:
                    optimizer = torch.optim.SGD([
                        {"params": [p for emb in dlrm.emb_sparse for p in emb.parameters()], "lr" : args.learning_rate / ext_dist.my_size},
                        {"params": [p for emb in dlrm.emb_dense for p in emb.parameters()], "lr" : args.learning_rate},
                        {"params": dlrm.bot_l.parameters(), "lr" : args.learning_rate},
                        {"params": dlrm.top_l.parameters(), "lr" : args.learning_rate}
                    ], lr=args.learning_rate)

            lr_scheduler = LRPolicyScheduler(optimizer, args.lr_num_warmup_steps, args.lr_decay_start_step,
                                             args.lr_num_decay_steps)
        else:
            optimizer = None
            lr_scheduler = None

        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train(self, nbatches=-1, nbatches_test=-1):
        args = self.args

        # training or inference
        best_gA_test = 0
        best_auc_test = 0
        skip_upto_epoch = 0
        skip_upto_batch = 0
        total_time = 0
        total_loss = 0
        total_accu = 0
        total_iter = 0
        total_samp = 0
        k = 0

        mlperf_logger.mlperf_submission_log('dlrm')
        mlperf_logger.log_event(key=mlperf_logger.constants.SEED, value=args.numpy_rand_seed)
        mlperf_logger.log_event(key=mlperf_logger.constants.GLOBAL_BATCH_SIZE, value=args.mini_batch_size)

        ext_dist.barrier()
        print("time/loss/accuracy (if enabled):")

        # LR is logged twice for now because of a compliance checker bug
        mlperf_logger.log_event(key=mlperf_logger.constants.OPT_BASE_LR, value=args.learning_rate)
        mlperf_logger.log_event(key=mlperf_logger.constants.OPT_LR_WARMUP_STEPS,
                                value=args.lr_num_warmup_steps)

        # use logging keys from the official HP table and not from the logging library
        mlperf_logger.log_event(key='sgd_opt_base_learning_rate', value=args.learning_rate)
        mlperf_logger.log_event(key='lr_decay_start_steps', value=args.lr_decay_start_step)
        mlperf_logger.log_event(key='sgd_opt_learning_rate_decay_steps', value=args.lr_num_decay_steps)
        mlperf_logger.log_event(key='sgd_opt_learning_rate_decay_poly_power', value=2)


        use_gpu = self.use_gpu
        use_ipex = args.use_ipex
        device = self.device
        train_ld = self.train_ds
        test_ld = self.test_ds
        with torch.autograd.profiler.profile(args.enable_profiling, use_gpu) as prof:
            while k < args.nepochs:
                mlperf_logger.barrier()
                mlperf_logger.log_start(key=mlperf_logger.constants.BLOCK_START,
                                        metadata={mlperf_logger.constants.FIRST_EPOCH_NUM: (k + 1),
                                                  mlperf_logger.constants.EPOCH_COUNT: 1})
                mlperf_logger.barrier()
                mlperf_logger.log_start(key=mlperf_logger.constants.EPOCH_START,
                                        metadata={mlperf_logger.constants.EPOCH_NUM: k + 1})

                if k < skip_upto_epoch:
                    continue

                accum_time_begin = self.time_wrap(use_gpu)

                if args.mlperf_logging:
                    previous_iteration_time = None

                for j, (X, lS_o, lS_i, T) in enumerate(train_ld):
                    if j == 0 and args.save_onnx:
                        (X_onnx, lS_o_onnx, lS_i_onnx) = (X, lS_o, lS_i)

                    if j < skip_upto_batch:
                        continue

                    if args.mlperf_logging:
                        current_time = self.time_wrap(use_gpu)
                        if previous_iteration_time:
                            iteration_time = current_time - previous_iteration_time
                        else:
                            iteration_time = 0
                        previous_iteration_time = current_time
                    else:
                        ext_dist.barrier()
                        t1 = self.time_wrap(use_gpu)

                    # early exit if nbatches was set by the user and has been exceeded
                    if nbatches > 0 and j >= nbatches:
                        break
                    '''
                    # debug prints
                    print("input and targets")
                    print(X.detach().cpu().numpy())
                    print([np.diff(S_o.detach().cpu().tolist()
                           + list(lS_i[i].shape)).tolist() for i, S_o in enumerate(lS_o)])
                    print([S_i.detach().cpu().numpy().tolist() for S_i in lS_i])
                    print(T.detach().cpu().numpy())
                    '''

                    # forward pass
                    Z = self.dlrm_wrap(X, lS_o, lS_i, use_gpu, use_ipex, device)

                    # loss
                    E = self.loss_fn_wrap(Z, T, use_gpu, use_ipex, device)
                    '''
                    # debug prints
                    print("output and loss")
                    print(Z.detach().cpu().numpy())
                    print(E.detach().cpu().numpy())
                    '''
                    # compute loss and accuracy
                    L = E.detach().cpu().numpy()  # numpy array
                    S = Z.detach().cpu().numpy()  # numpy array
                    T = T.detach().cpu().numpy()  # numpy array
                    mbs = T.shape[0]  # = args.mini_batch_size except maybe for last
                    A = np.sum((np.round(S, 0) == T).astype(np.uint8))

                    if not args.inference_only:
                        # scaled error gradient propagation
                        # (where we do not accumulate gradients across mini-batches)
                        self.optimizer.zero_grad()
                        # backward pass
                        E.backward()
                        # debug prints (check gradient norm)
                        # for l in mlp.layers:
                        #     if hasattr(l, 'weight'):
                        #          print(l.weight.grad.norm().item())

                        # optimizer
                        self.optimizer.step()
                        self.lr_scheduler.step()

                    if args.mlperf_logging:
                        total_time += iteration_time
                    else:
                        t2 = self.time_wrap(use_gpu)
                        total_time += t2 - t1
                    total_accu += A
                    total_loss += L * mbs
                    total_iter += 1
                    total_samp += mbs

                    should_print = ((j + 1) % args.print_freq == 0) or (j + 1 == nbatches)
                    should_test = (
                            (args.test_freq > 0)
                            and (args.data_generation == "dataset")
                            and (((j + 1) % args.test_freq == 0) or (j + 1 == nbatches))
                    )

                    # print time, loss and accuracy
                    if should_print or should_test:
                        gT = 1000.0 * total_time / total_iter if args.print_time else -1
                        total_time = 0

                        gA = total_accu / total_samp
                        total_accu = 0

                        gL = total_loss / total_samp
                        total_loss = 0

                        str_run_type = "inference" if args.inference_only else "training"
                        print(
                            "Finished {} it {}/{} of epoch {}, {:.2f} ms/it, ".format(
                                str_run_type, j + 1, nbatches, k, gT
                            )
                            + "loss {:.6f}, accuracy {:3.3f} %".format(gL, gA * 100)
                        )
                        # Uncomment the line below to print out the total time with overhead
                        # print("Accumulated time so far: {}" \
                        # .format(time_wrap(use_gpu) - accum_time_begin))
                        total_iter = 0
                        total_samp = 0

                    # testing
                    if should_test and not args.inference_only:
                        epoch_num_float = (j + 1) / len(train_ld) + k + 1
                        mlperf_logger.barrier()
                        mlperf_logger.log_start(key=mlperf_logger.constants.EVAL_START,
                                                metadata={mlperf_logger.constants.EPOCH_NUM: epoch_num_float})

                        # don't measure training iter time in a test iteration
                        if args.mlperf_logging:
                            previous_iteration_time = None

                        test_accu = 0
                        test_loss = 0
                        test_samp = 0

                        accum_test_time_begin = self.time_wrap(use_gpu)
                        if args.mlperf_logging:
                            scores = []
                            targets = []

                        for i, (X_test, lS_o_test, lS_i_test, T_test) in enumerate(test_ld):
                            # early exit if nbatches was set by the user and was exceeded
                            if nbatches > 0 and i >= nbatches:
                                break

                            t1_test = self.time_wrap(use_gpu)

                            # forward pass
                            Z_test = self.dlrm_wrap(
                                X_test, lS_o_test, lS_i_test, use_gpu, use_ipex, device
                            )
                            if args.mlperf_logging:
                                if ext_dist.my_size > 1:
                                    Z_test = ext_dist.all_gather(Z_test, None)
                                    T_test = ext_dist.all_gather(T_test, None)
                                S_test = Z_test.detach().cpu().numpy()  # numpy array
                                T_test = T_test.detach().cpu().numpy()  # numpy array
                                scores.append(S_test)
                                targets.append(T_test)
                            else:
                                # loss
                                E_test = self.loss_fn_wrap(Z_test, T_test, use_gpu, use_ipex, device)

                                # compute loss and accuracy
                                L_test = E_test.detach().cpu().numpy()  # numpy array
                                S_test = Z_test.detach().cpu().numpy()  # numpy array
                                T_test = T_test.detach().cpu().numpy()  # numpy array
                                mbs_test = T_test.shape[0]  # = mini_batch_size except last
                                A_test = np.sum((np.round(S_test, 0) == T_test).astype(np.uint8))
                                test_accu += A_test
                                test_loss += L_test * mbs_test
                                test_samp += mbs_test

                            t2_test = self.time_wrap(use_gpu)

                        if args.mlperf_logging:
                            scores = np.concatenate(scores, axis=0)
                            targets = np.concatenate(targets, axis=0)

                            validation_results = {}
                            if args.use_ipex:
                                validation_results['roc_auc'], validation_results['loss'], validation_results['accuracy'] = \
                                    core.roc_auc_score(torch.from_numpy(targets).reshape(-1), torch.from_numpy(scores).reshape(-1))
                            else:
                                metrics = {
                                    'loss' : sklearn.metrics.log_loss,
                                    'recall' : lambda y_true, y_score:
                                    sklearn.metrics.recall_score(
                                        y_true=y_true,
                                        y_pred=np.round(y_score)
                                    ),
                                    'precision' : lambda y_true, y_score:
                                    sklearn.metrics.precision_score(
                                        y_true=y_true,
                                        y_pred=np.round(y_score)
                                    ),
                                    'f1' : lambda y_true, y_score:
                                    sklearn.metrics.f1_score(
                                        y_true=y_true,
                                        y_pred=np.round(y_score)
                                    ),
                                    'ap' : sklearn.metrics.average_precision_score,
                                    'roc_auc' : sklearn.metrics.roc_auc_score,
                                    'accuracy' : lambda y_true, y_score:
                                    sklearn.metrics.accuracy_score(
                                        y_true=y_true,
                                        y_pred=np.round(y_score)
                                    ),
                                }

                                # print("Compute time for validation metric : ", end="")
                                # first_it = True
                                for metric_name, metric_function in metrics.items():
                                    # if first_it:
                                    #     first_it = False
                                    # else:
                                    #     print(", ", end="")
                                    # metric_compute_start = time_wrap(False)
                                    validation_results[metric_name] = metric_function(
                                        targets,
                                        scores
                                    )
                                    # metric_compute_end = time_wrap(False)
                                    # met_time = metric_compute_end - metric_compute_start
                                    # print("{} {:.4f}".format(metric_name, 1000 * (met_time)),
                                    #      end="")

                            # print(" ms")
                            gA_test = validation_results['accuracy']
                            gL_test = validation_results['loss']
                        else:
                            gA_test = test_accu / test_samp
                            gL_test = test_loss / test_samp

                        is_best = gA_test > best_gA_test
                        if is_best:
                            best_gA_test = gA_test
                            if not (args.save_model == ""):
                                print("Saving model to {}".format(args.save_model))
                                torch.save(
                                    {
                                        "epoch": k,
                                        "nepochs": args.nepochs,
                                        "nbatches": nbatches,
                                        "nbatches_test": nbatches_test,
                                        "iter": j + 1,
                                        "state_dict": self.dlrm_net.state_dict(),
                                        "train_acc": gA,
                                        "train_loss": gL,
                                        "test_acc": gA_test,
                                        "test_loss": gL_test,
                                        "total_loss": total_loss,
                                        "total_accu": total_accu,
                                        "opt_state_dict": self.optimizer.state_dict(),
                                    },
                                    args.save_model,
                                )

                        if args.mlperf_logging:
                            is_best = validation_results['roc_auc'] > best_auc_test
                            if is_best:
                                best_auc_test = validation_results['roc_auc']

                            mlperf_logger.log_event(key=mlperf_logger.constants.EVAL_ACCURACY,
                                                    value=float(validation_results['roc_auc']),
                                                    metadata={mlperf_logger.constants.EPOCH_NUM: epoch_num_float})
                            print(
                                "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, k)
                                + " loss {:.6f},".format(
                                    validation_results['loss']
                                )
                                + " auc {:.4f}, best auc {:.4f},".format(
                                    validation_results['roc_auc'],
                                    best_auc_test
                                )
                                + " accuracy {:3.3f} %, best accuracy {:3.3f} %".format(
                                    validation_results['accuracy'] * 100,
                                    best_gA_test * 100
                                )
                            )
                        else:
                            print(
                                "Testing at - {}/{} of epoch {},".format(j + 1, nbatches, 0)
                                + " loss {:.6f}, accuracy {:3.3f} %, best {:3.3f} %".format(
                                    gL_test, gA_test * 100, best_gA_test * 100
                                )
                            )
                        mlperf_logger.barrier()
                        mlperf_logger.log_end(key=mlperf_logger.constants.EVAL_STOP,
                                              metadata={mlperf_logger.constants.EPOCH_NUM: epoch_num_float})

                        # Uncomment the line below to print out the total time with overhead
                        # print("Total test time for this group: {}" \
                        # .format(time_wrap(use_gpu) - accum_test_time_begin))

                        if (args.mlperf_logging
                                and (args.mlperf_acc_threshold > 0)
                                and (best_gA_test > args.mlperf_acc_threshold)):
                            print("MLPerf testing accuracy threshold "
                                  + str(args.mlperf_acc_threshold)
                                  + " reached, stop training")
                            break

                        if (args.mlperf_logging
                                and (args.mlperf_auc_threshold > 0)
                                and (best_auc_test > args.mlperf_auc_threshold)):
                            print("MLPerf testing auc threshold "
                                  + str(args.mlperf_auc_threshold)
                                  + " reached, stop training")
                            mlperf_logger.barrier()
                            mlperf_logger.log_end(key=mlperf_logger.constants.RUN_STOP,
                                                  metadata={
                                                      mlperf_logger.constants.STATUS: mlperf_logger.constants.SUCCESS})
                            break

                mlperf_logger.barrier()
                mlperf_logger.log_end(key=mlperf_logger.constants.EPOCH_STOP,
                                      metadata={mlperf_logger.constants.EPOCH_NUM: k + 1})
                mlperf_logger.barrier()
                mlperf_logger.log_end(key=mlperf_logger.constants.BLOCK_STOP,
                                      metadata={mlperf_logger.constants.FIRST_EPOCH_NUM: k + 1})
                k += 1  # nepochs

        if args.mlperf_logging and best_auc_test <= args.mlperf_auc_threshold:
            mlperf_logger.barrier()
            mlperf_logger.log_end(key=mlperf_logger.constants.RUN_STOP,
                                  metadata={mlperf_logger.constants.STATUS: mlperf_logger.constants.ABORTED})

        # profiling
        if args.enable_profiling:
            with open("dlrm_s_pytorch.prof", "w") as prof_f:
                prof_f.write(prof.key_averages().table(sort_by="cpu_time_total"))
                prof.export_chrome_trace(f"./dlrm_s_pytorch_{self.world_rank}.json")
            # print(prof.key_averages().table(sort_by="cpu_time_total"))

        # test prints
        if not args.inference_only and args.debug_mode:
            print("updated parameters (weights and bias):")
            for param in self.dlrm_net.parameters():
                print(param.detach().cpu().numpy())


class DLRMRunArguments:
    def __init__(self, ray_address: str,
                 ray_redis_password: str,
                 ray_node_ip_address: str,
                 spark_app_name: str,
                 spark_num_executors: int,
                 spark_executor_cores: int,
                 spark_executor_memory: str,
                 spark_extra_config: Dict,
                 pre_proc_args: PreProcArguments,
                 model_args: ModelArguments):

        # ray arguments
        self.ray_address = ray_address
        self.ray_redis_password = ray_redis_password
        self.ray_node_ip_address = ray_node_ip_address

        # spark arguments
        self.spark_app_name = spark_app_name
        self.spark_num_executors = spark_num_executors
        self.spark_executor_cores = spark_executor_cores
        self.spark_executor_memory = spark_executor_memory
        self.spark_extra_config = spark_extra_config

        # preprocess arguments
        self.pre_proc_args = pre_proc_args

        # model arguments
        self.model_args = model_args


def collate_fn(
        df: pd.DataFrame,
        num_numerical_features: int,
        max_ind_range: int,
        flag_input_torch_tensor=False):
    array = df.values
    x_int_batch = array[:, 1:1 + num_numerical_features].view(dtype=np.float32)
    x_cat_batch = array[:, 1 + num_numerical_features:]
    y_batch = array[:, 0]

    if max_ind_range > 0:
        x_cat_batch = x_cat_batch % max_ind_range

    if flag_input_torch_tensor:
        x_int_batch = torch.log(x_int_batch.clone().detach().type(torch.float) + 1)
        x_cat_batch = x_cat_batch.clone().detach().type(torch.long)
        y_batch = y_batch.clone().detach().type(torch.float32).view(-1, 1)
    else:
        x_int_batch = torch.log(torch.tensor(x_int_batch, dtype=torch.float) + 1)
        x_cat_batch = torch.tensor(x_cat_batch, dtype=torch.long)
        y_batch = torch.tensor(y_batch, dtype=torch.float32).view(-1, 1)

    batch_size = x_cat_batch.shape[0]
    feature_count = x_cat_batch.shape[1]
    lS_o = torch.arange(batch_size).reshape(1, -1).repeat(feature_count, 1)

    return x_int_batch, lS_o, x_cat_batch.t(), y_batch.view(-1, 1)


def create_torch_ds(ds: ml_data.MLDataset,
                    num_shards: int,
                    shard_index: int,
                    batch_size: int,
                    shuffle: bool):
    assert shard_index < num_shards
    shard_ids = []
    i = shard_index
    step = num_shards
    while i < ds.num_shards():
        shard_ids.append(i)
        i += step
    ds = ds.select_shards(shard_ids)
    return TorchDataset(ds, batch_size, shuffle)


class TorchDataset(IterableDataset):
    def __init__(self,
                 ds: ml_data.MLDataset,
                 batch_size: int,
                 shuffle: bool):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        it = self.ds.gather_async(batch_ms=0, num_async=self.ds.num_shards())
        it = iter(it)
        return_df = None
        while True:
            try:
                cur_df = next(it)
                cur_index = 0
                cur_size = cur_df.shape[0]
                while cur_df is not None or (
                        cur_index + self.batch_size) < cur_size:
                    if cur_df is None or cur_index == cur_size:
                        cur_df = next(it)
                        cur_index = 0
                        cur_size = cur_df.shape[0]
                    if return_df is not None:
                        ri = cur_index + self.batch_size - return_df.shape[0]
                        ri = min(ri, cur_size)
                        tmp = cur_df.iloc[cur_index:ri]
                        return_df = pd.concat([return_df, tmp])
                        cur_index = ri
                    else:
                        ri = cur_index + self.batch_size
                        ri = min(ri, cur_size)
                        return_df = cur_df.iloc[cur_index:ri]
                        cur_index = ri
                    if return_df.shape[0] == self.batch_size:
                        if self.shuffle:
                            return_df = return_df.sample(frac=1)
                        yield return_df
                        return_df = None
            except StopIteration:
                break

        if return_df is not None:
            if self.shuffle:
                return_df = return_df.sample(frac=1)
            yield return_df


def dlrm_run(num_workers: int,
             multiple_for_mlds: int,
             cores_per_worker: int,
             inter_op_parallelism: int,
             intra_op_parallelism: int,
             args: DLRMRunArguments):

    assert cores_per_worker == (inter_op_parallelism * intra_op_parallelism)

    # connect to ray
    ray.init(address=args.ray_address,
             _redis_password=args.ray_redis_password,
             _node_ip_address=args.ray_node_ip_address)

    try:
        # start up spark cluster
        spark = raydp.init_spark(app_name=args.spark_app_name,
                                 num_executors=args.spark_num_executors,
                                 executor_cores=args.spark_executor_cores,
                                 executor_memory=args.spark_executor_memory,
                                 configs=args.spark_extra_config)
        # preprocessing
        train_folder, test_folder, model_size = pre_process(spark, args.pre_proc_args)
    except:
        raise
    finally:
        raydp.stop_spark()

    train_ds = ml_data.read_parquet(train_folder, num_workers * multiple_for_mlds)
    # test_ds = ml_data.read_parquet(train_folder, num_workers * multiple_for_mlds)

    # set up cluster for distributed training
    executors = []
    for i in range(num_workers):
        executors.append(Executor.options(num_cpus=cores_per_worker).remote(inter_op_parallelism,
                                                                            intra_op_parallelism,
                                                                            num_workers,
                                                                            args.model_args))

    # get master address and port
    master_addr, master_port = ray.get(executors[0].setup_address.remote())

    # set up process group
    ray.get([executor.setup_process_group.remote(master_addr, master_port, i)
             for i, executor in enumerate(executors)])

    # set up dataset
    ids = []
    for i, executor in enumerate(executors):
        train_dataset = create_torch_ds(train_ds, num_workers, i,
                                        args.model_args.mini_batch_size,
                                        args.model_args.mlperf_bin_shuffle)
        test_dataset = None
        ids.append(executor.set_dataset.remote(train_dataset, test_dataset))
    ray.get(ids)

    # create model
    ln_emb = list(model_size.values())
    ray.get([executor.create_model.remote(ln_emb) for executor in executors])

    # train model
    nbatches = -1
    nbatches_test = -1
    ray.get([executor.train.remote(nbatches, nbatches_test) for executor in executors])
