import functools
import time
from collections import OrderedDict

import pandas as pd
import ray.util.data as ml_data
import raydp
import torch
import yaml
from raydp.mpi import create_mpi_job, MPIJobContext
from raydp.spark import RayMLDataset

from dlrm.preproc.spark_preproc import transform, generate_models, PreProcArguments
from dlrm.utils import import_model_arguments

ModelArguments = import_model_arguments()

columns = [f"_c{i}" for i in range(40)]


def collate_fn(
        df: pd.DataFrame,
        num_numerical_features: int,
        max_ind_range: int,
        flag_input_torch_tensor=False):
    df = df[columns]
    # array = df.values
    # x_int_batch = array[:, 1:1 + num_numerical_features].astype(dtype=np.float32)
    # x_cat_batch = array[:, 1 + num_numerical_features:].astype(dtype=np.int64)
    # y_batch = array[:, 0].astype(dtype=np.float32)
    # x_int_batch = torch.from_numpy(x_int_batch)
    # x_cat_batch = torch.from_numpy(x_cat_batch)
    # y_batch = torch.from_numpy(y_batch)
    tensor = torch.from_numpy(df.values).view((-1, 40))
    x_int_batch = tensor[:, 1:1 + num_numerical_features]
    x_cat_batch = tensor[:, 1 + num_numerical_features:]
    y_batch = tensor[:, 0]

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


def data_preprocess(
        app_name: str = None,
        num_executors: int = None,
        executor_cores: int = None,
        executor_memory: str = None,
        configs=None,
        args: PreProcArguments = None,
        return_df: bool = False):
    try:
        start = time.time()
        spark = raydp.init_spark(
            app_name=app_name,
            num_executors=num_executors,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            configs=configs)

        # generate models
        generate_models(spark, args, args.total_days_range)
        args.low_mem = False

        # transform for train data
        train_data, model_size = transform(spark, args, args.train_days_range, True, return_df)

        # transform for test data
        args.output_ordering = "input"
        test_data, _ = transform(spark, args, args.test_days_range, False, return_df)

        print("Data preprocessing duration:", time.time() - start)

        return train_data, test_data, model_size
    except:
        raise
    finally:
        if not return_df:
            raydp.stop_spark()


def create_train_task(
        num_workers: int = None,
        train_ds: ml_data.MLDataset = None,
        test_ds: ml_data.MLDataset = None,
        model_size=None,
        args: ModelArguments = None,
        nbatches: int = -1,
        nbatches_test: int = -1):

    fn = functools.partial(collate_fn,
                           num_numerical_features=args.arch_dense_feature_size,
                           max_ind_range=args.max_ind_range,
                           flag_input_torch_tensor=True)

    def task_fn(context):
        batch_size = args.mini_batch_size // num_workers
        train_loader = RayMLDataset.to_torch(ds=train_ds,
                                             world_size=num_workers,
                                             world_rank=context.world_rank,
                                             batch_size=batch_size,
                                             collate_fn=fn,
                                             shuffle=True,
                                             shuffle_seed=int(time.time()),
                                             local_rank=context.local_rank,
                                             prefer_node=context.node_ip,
                                             prefetch=True)

        test_batch_size = args.test_mini_batch_size // num_workers
        test_loader = RayMLDataset.to_torch(ds=test_ds,
                                            world_size=num_workers,
                                            world_rank=context.world_rank,
                                            batch_size=test_batch_size,
                                            collate_fn=fn,
                                            shuffle=True,
                                            shuffle_seed=int(time.time()),
                                            local_rank=context.local_rank,
                                            prefer_node=context.node_ip,
                                            prefetch=True)

        from dlrm.utils import import_model_training
        model_training = import_model_training()
        try:
            model_training(args, train_loader, test_loader, model_size,
                           nbatches, nbatches_test)
        except Exception as e:
            time.sleep(1)
            print(e, flush=True)
        finally:
            return context.world_rank
    return task_fn


def run(config_path: str):
    shuffle_seed = int(time.time())

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # data preprocess
    data_config = config["data_preprocess"]
    pre_proc_args = PreProcArguments(total_days=data_config["total_days"],
                                     train_days=data_config["train_days"],
                                     test_days=data_config["test_days"],
                                     input_folder=data_config["input_folder"],
                                     test_input_folder=data_config["test_input_folder"],
                                     output_folder=data_config["output_folder"],
                                     model_size_file=data_config["model_size_file"],
                                     model_folder=data_config["model_folder"],
                                     write_mode=data_config["write_mode"],
                                     dict_build_shuffle_parallel_per_day=data_config["dict_build_shuffle_parallel_per_day"],
                                     low_mem=data_config["low_mem"],
                                     frequency_limit=str(data_config["frequency_limit"]))

    start = time.time()
    train_data, test_data, model_size = data_preprocess(app_name=data_config["app_name"],
                                                        num_executors=data_config["num_executors"],
                                                        executor_cores=data_config["executor_cores"],
                                                        executor_memory=data_config["executor_memory"],
                                                        configs=data_config["spark_config"]["configs"],
                                                        args=pre_proc_args,
                                                        return_df=False)
    model_size = OrderedDict([(key, value + 1) for key, value in model_size.items()])
    print("Data dreprocess duration: ", time.time() - start)
    start = time.time()

    mpi_config = config["mpi_config"]
    mpi_env = config["env"]

    def mpi_script_prepare_fn(context: MPIJobContext):
        extra_env = {}
        extra_env["http_proxy"] = ""
        extra_env["https_proxy"] = ""

        for k in mpi_env:
            extra_env[k] = str(mpi_env[k])

        scripts = []
        scripts.append("mpiexec.hydra")
        scripts.append("-genv")
        scripts.append(str(mpi_config["I_MPI_PIN_DOMAIN"]))
        scripts.append("-hosts")
        scripts.append(",".join(context.hosts))
        scripts.append("-ppn")
        scripts.append(str(context.num_procs_per_node))
        scripts.append("-prepend-rank")
        return scripts

    job = create_mpi_job(job_name=mpi_config["job_name"],
                         world_size=mpi_config["world_size"],
                         num_cpus_per_process=mpi_config["num_cpus_per_process"],
                         num_processes_per_node=mpi_config["num_processes_per_node"],
                         mpi_script_prepare_fn=mpi_script_prepare_fn,
                         timeout=mpi_config["timeout"],
                         mpi_type=mpi_config["mpi_type"])

    addresses = job.get_rank_addresses()

    def set_master_addr(context):
        import os
        os.environ["MASTER_ADDR"] = str(addresses[0])
        return context.world_rank

    job.run(set_master_addr)

    num_shards = mpi_config["world_size"]
    train_ds = RayMLDataset.from_parquet(paths=train_data,
                                         num_shards=num_shards,
                                         shuffle=True,
                                         shuffle_seed=shuffle_seed,
                                         node_hints=addresses)
    test_ds = RayMLDataset.from_parquet(paths=test_data,
                                        num_shards=num_shards,
                                        shuffle=True,
                                        shuffle_seed=shuffle_seed,
                                        node_hints=addresses)

    model_config = config["model_arguments"]
    model_args = ModelArguments(dist_backend=model_config["dist_backend"],
                                sparse_dense_boundary=model_config["sparse_dense_boundary"],
                                bf16=model_config["bf16"],
                                use_ipex=model_config["use_ipex"],
                                arch_dense_feature_size=model_config["arch_dense_feature_size"],
                                arch_sparse_feature_size=model_config["arch_sparse_feature_size"],
                                arch_mlp_bot=model_config["arch_mlp_bot"],
                                arch_mlp_top=model_config["arch_mlp_top"],
                                max_ind_range=model_config["max_ind_range"],
                                loss_function=model_config["loss_function"],
                                round_targets=model_config["round_targets"],
                                learning_rate=model_config["learning_rate"],
                                mini_batch_size=model_config["mini_batch_size"],
                                test_mini_batch_size=model_config["test_mini_batch_size"],
                                print_freq=model_config["print_freq"],
                                print_time=model_config["print_time"],
                                test_freq=model_config["test_freq"],
                                lr_num_warmup_steps=model_config["lr_num_warmup_steps"],
                                lr_decay_start_step=model_config["lr_decay_start_step"],
                                lr_num_decay_steps=model_config["lr_num_decay_steps"],
                                mlperf_logging=model_config["mlperf_logging"],
                                mlperf_auc_threshold=model_config["mlperf_auc_threshold"],
                                mlperf_bin_shuffle=model_config["mlperf_bin_shuffle"],
                                numpy_rand_seed=model_config["numpy_rand_seed"],
                                data_generation=model_config["data_generation"],
                                data_set=model_config["data_set"],
                                memory_map=model_config["memory_map"],
                                processed_data_file=model_config["processed_data_file"],
                                mlperf_bin_loader=model_config["mlperf_bin_loader"],
                                enable_profiling=model_config["enable_profiling"],
                                nepochs=model_config["nepochs"])

    train_fn = create_train_task(
        num_workers=num_shards,
        train_ds=train_ds,
        test_ds=test_ds,
        model_size=model_size,
        args=model_args)
    job.run(train_fn)
    job.stop()
    print("Model training duration: ", time.time() - start)
