import functools
import time
from typing import List

import pandas as pd
import ray
import ray.util.data as ml_data
import ray.util.iter as parallel_it
import raydp
import torch
from pyspark.sql.dataframe import DataFrame
from raydp.spark.dataset import _save_spark_df_to_object_store
from torch.utils.data.dataset import IterableDataset

from .dlrm_s_pytorch import ModelArguments
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


def create_torch_ds(ds: ml_data.MLDataset,
                    num_shards: int,
                    shard_index: int,
                    batch_size: int,
                    shuffle: bool,
                    collate_fn):
    assert shard_index < num_shards
    shard_ids = []
    i = shard_index
    step = num_shards
    while i < ds.num_shards():
        shard_ids.append(i)
        i += step
    ds = ds.select_shards(shard_ids)
    return TorchDataset(ds, batch_size // num_shards, shuffle, collate_fn)


class TorchDataset(IterableDataset):
    def __init__(self,
                 ds: ml_data.MLDataset,
                 batch_size: int,
                 shuffle: bool,
                 collate_fn):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn

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
                        yield self.collate_fn(return_df)
                        return_df = None
            except StopIteration:
                break

        if return_df is not None:
            if self.shuffle:
                return_df = return_df.sample(frac=1)
            yield self.collate_fn(return_df)


def preprocess_generate_models(
        app_name: str = None,
        num_executors: int = None,
        executor_cores: int = None,
        executor_memory: str = None,
        configs=None,
        args: PreProcArguments = None):
    try:
        spark = raydp.init_spark(
            app_name=app_name,
            num_executors=num_executors,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            configs=configs)

        generate_models(spark, args, args.total_days_range)
    except:
        raise
    finally:
        raydp.stop_spark()


def preprocess_transform(
        app_name: str = None,
        num_executors: int = None,
        executor_cores: int = None,
        executor_memory: str = None,
        configs=None,
        args: PreProcArguments = None,
        return_df: bool = False):
    try:
        spark = raydp.init_spark(
            app_name=app_name,
            num_executors=num_executors,
            executor_cores=executor_cores,
            executor_memory=executor_memory,
            configs=configs)
        # transform for train data
        train_data, model_size = transform(spark, args, args.train_days_range, True, return_df)

        # transform for test data
        args.low_mem = False
        test_data, val_data, _ = transform(spark, args, args.test_days_range, False, return_df)

        return train_data, test_data, val_data, model_size
    except:
        raise
    finally:
        if not return_df:
            raydp.stop_spark()


def create_ml_dataset_from_spark(df: DataFrame,
                                 num_workers: int,
                                 actors: List[ray.actor.ActorHandle]):
    start = time.time()
    assert num_workers == len(actors)
    df = df.repartition(num_workers)
    record_batch_set = _save_spark_df_to_object_store(df, num_workers, True)
    ray.get([actor.init.remote(batch, False) for actor, batch in zip(actors, record_batch_set)])
    it = parallel_it.from_actors(actors, "DataFrame MLDataset")
    ds = ml_data.from_parallel_iter(it, need_convert=False, batch_size=0, repeated=False)
    print(f"Duration: {time.time() - start}")
    return ds


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
            train_dataset = create_torch_ds(train_ds, num_workers, context.world_rank,
                args.mini_batch_size, args.mlperf_bin_shuffle, fn)
        
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=None,
                batch_sampler=None,
                shuffle=False,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
                sampler=None
        )

        test_loader = None
        if test_ds is not None:
            test_dataset = create_torch_ds(test_ds, num_workers, context.world_rank,
                args.test_mini_batch_size, False, fn)
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=None,
                batch_sampler=None,
                shuffle=False,
                num_workers=0,
                collate_fn=None,
                pin_memory=False,
                drop_last=False,
                sampler=None
            )
        from dlrm.dlrm_s_pytorch import model_training
        try:
            model_training(args, train_loader, test_loader, model_size,
                nbatches, nbatches_test)
        except Exception as e:
            time.sleep(1)
            print(e, flush=True)
        finally:
            return context.world_rank
    return task_fn
