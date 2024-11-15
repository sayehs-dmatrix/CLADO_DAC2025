# Source code for CLADO
We provide our code implementation of CLADO. We also include implementations of MPQCO and HAWQV3. 

**Structure of codes:** All codes including three Python files and one bash script are placed under the _main_ folder. 
- _prep_mpqco_clado.py_: computes the sensitivities for MPQCO and CLADO. 
- _prep_hawq.py_: computes the Hessian traces, which are used later by optimize.py to compute the sensitivities of HAWQ. 
- _optimize.py_: takes the pre-computed sensitivities (traces for HAWQ) and solves the corresponding IQP(for CLADO)/ILP(for HAWQ and MPQCO) problems to get the MPQ decisions. It then evaluates the decisions and report quantized models’ performance.

**Dependencies:** To run the codes, a CUDA environment with PyTorch, MQBench, Pyhessian, and CVXPY (with GUROBI backend) packages is required. We recommend to use PyTorch==1.10.1, MQBench==0.0.6, CVXPY==1.2.1, Pyhessian==0.1, and GUROBI-PY==9.5.2 to avoid any compatibility issues.

**MPQ sample runs:** sample run.sh is an example script to run the experiments. It launches a quick run of three algorithms on the ResNet-34 model using a randomly sampled 64-sample sensitivity set. A datapath to the ImageNet dataset is required, it needs to be specified by user through the dp variable in the script. Once finished, one can check optimize.log for results. By default, we use bs=64 for batch size. The sb and eb define the indices of the starting and ending (inclusive) batches to be included in the sensitivity set (e.g., with the default batch size, one can specify sb=0,eb=15 to include 1024 samples in the sensitivity set). modelname specifies the model name of pretrained models by PyTorch, one can change it to other valid names (e.g., “resnet50”) listed in https://pytorch.org/vision/stable/models.html. For other program arguments and hyperparameters, (e.g., CUDA device, number of threads, sampling of sensitivity samples, model size calculations and constraints), please see the codes for more details. Note that HAWQ uses Hutchinson’s method (via Pyhessian) to compute the Hessian traces, which is demanding of GPU memory. Therefore, when experimenting with deep models (e.g., ResNet-50), one may use a smaller batch size than 64 to avoid the OOM error.

**Quantization settings:** The quantization of models are handled by the MQBench package. The details of MQBench’s quantizer settings can be found in _prep_mpqco_clado.py_ (line 154 − 181). One can modify the
MQBench settings there to test other settings of interest by specifying different hyperparameters including quantization granularity (per-channel or per-tensor), algorithms for quantization scale factors (MSE or Min-Max), formats of scale factors (power of two or continuous), etc.
