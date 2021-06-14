## Install DLRM related library
https://github.com/mlperf/training_results_v0.7/tree/master/Intel/benchmarks/dlrm/1-node-8s-cpx-2-pytorch

## Install Ray and RayDP:
Ray version: >= 1.3.0 and should include this patch https://github.com/ray-project/ray/pull/16045
RayDP: latest code
dlrm: python setup.py bdist_wheel && pip install dist/dlrm-*.whl

## Running:
We use the mlperf code by default, you should set env: `use_dlrm_optimized=True` before import dlrm if you want to use the optimized code from Jian's team.

```python

import dlrm

dlrm.run("config file path")
```

