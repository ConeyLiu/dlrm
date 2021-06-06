from .preproc.spark_preproc import PreProcArguments
from .main import data_preprocess, create_train_task

import os

if "use_dlrm_optimized" in os.environ:
    from .dlrm_s_pytorch_optimized import ModelArguments
else:
    from .dlrm_s_pytorch_original import ModelArguments
