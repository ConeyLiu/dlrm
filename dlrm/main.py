import functools
import queue
import threading
import time
from typing import List

import pandas as pd
import ray.util.data as ml_data
import raydp
import torch

from raydp.torch import create_data_loader
from torch.utils.data.dataset import IterableDataset

import os

if "use_dlrm_optimized" in os.environ:
    from .dlrm_s_pytorch_optimized import ModelArguments
else:
    from .dlrm_s_pytorch_original import ModelArguments
from .preproc.spark_preproc import transform, generate_models, PreProcArguments

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
        train_loader = None
        if train_ds is not None:
            batch_size = args.mini_batch_size // num_workers
            train_loader = create_data_loader(train_ds, num_workers, context.world_rank,
                context.local_rank, batch_size, fn, True, int(time.time()), context.node_ip, True)

        test_loader = None
        if test_ds is not None:
            test_batch_size = args.test_mini_batch_size // num_workers
            test_loader = create_data_loader(test_ds, num_workers, context.world_rank,
                context.local_rank, test_batch_size, fn, False, None, context.node_ip, True)
        from dlrm.dlrm_s_pytorch_original import model_training
        try:
            model_training(args, train_loader, test_loader, model_size,
                nbatches, nbatches_test)
        except Exception as e:
            time.sleep(1)
            print(e, flush=True)
        finally:
            return context.world_rank
    return task_fn
