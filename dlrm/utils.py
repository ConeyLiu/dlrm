import os


def import_model_arguments():
    if "use_dlrm_optimized" in os.environ:
        from .dlrm_s_pytorch_optimized import ModelArguments
    else:
        from .dlrm_s_pytorch_original import ModelArguments
    return ModelArguments


def import_model_training():
    if "use_dlrm_optimized" in os.environ:
        from .dlrm_s_pytorch_optimized import model_training
    else:
        from .dlrm_s_pytorch_original import model_training
    return model_training
