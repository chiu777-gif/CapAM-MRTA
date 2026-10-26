Learning Scalable Policies over Graphs for Multi-Robot Task Allocation using Capsule Attention Networks
(CapAM)

[中文使用说明书：安装、训练、评估、命令与参数说明](README.zh-CN.md)

The current entry points separate uniformly generated training scenarios from fixed test data.
Environment arguments use physical paper units: `--agent_max_speed 1` is converted
to `0.01` together with coordinates divided by 100. See the Chinese manual for the
updated data format, checkpoint compatibility, and complete examples.

Select `--encoder gcapcn` (default, one capsule layer) or `--encoder am`
(three self-attention layers by default; configurable with `--n_encode_layers`).
`--encoder mamba` selects a pure PyTorch Mamba-1 encoder (three layers by default,
RMSNorm, state dimension 8, expansion 1, convolution width 2). Its MRTA adapter
sorts tasks by coordinates and deadline, then restores the original task IDs;
duplicate feature vectors share a pooled representation. The reference scan
does not use optimized CUDA kernels and makes no throughput guarantee.
All three share the MRTA decoder and training objective. Evaluation with `eval.py` or
`z_test/evaluate.py` reads the encoder from the saved `args.json`; existing
checkpoints without this field continue to load as GCAPCN.

```bash
python run.py --graph_size 10 --n_agents 2 --batch_size 2 --epoch_size 4 --val_size 4 --eval_batch_size 2 --n_epochs 1 --no_cuda
python generate_mrta_data.py --graph_sizes 10 --n_agents 2 --dataset_size 8 --filename data/custom/uniform_10.pkl
python eval.py --model outputs/mrta_10/REPLACE_WITH_RUN_DIRECTORY --datasets data/custom/uniform_10.pkl --no_cuda -o results/uniform_10.json
```

This repository contains the code for the implementation of Capsule based attention networks aka CapAM, for solving a Single-Task Robot Single Robot-Task (SR-ST) Multi-Robot Task Allocation problem.
Environment:
    The SR-ST MRTA environment can be initialized using the MRTA class inside problem/mrta/problem_mrta.py. All the state space variables and the necessary functions
    to update the state space are implemented using the StateMRTA class inside problem/mrta/state_mrta.py
The train.py implements a REINFORCE algorithm

![](gif_latest.gif)

How to use the code:

    Training:
        To start training run the run.py file. The options.py can be used for different environement settings such as number of taks, number of robots
        and other parameters.
        The trained models for each epoch will be stored in a directory named 'outputs'
        We recommend using a GPU for training.

    Evaluation:
        Run eval.py with --model and --datasets. Task and robot counts come from the fixed input files.
        Use data/mrta_gini to load the original 96 TAPTC text cases directly, or generate versioned data as shown above.
        The old tuple datasets under data/mrta are incompatible with the current workload/robot schema.

    Test data generation:
        More test data with varying number of taks can be generated using the script generate_mrta_data.py

Dependencies:
    The code is written in Python. We recommend using Python 3.7. The required packages can be found in the file environment.yml.

Citation:

    Please cite our work if you find it useful.
S. Paul, P. Ghassemi and S. Chowdhury, "Learning Scalable Policies over Graphs for Multi-Robot Task Allocation using Capsule Attention Networks," 2022 International Conference on Robotics and Automation (ICRA), 2022, pp. 8815-8822, doi: 10.1109/ICRA46639.2022.9812370.

    
    Bibtex: 
           @INPROCEEDINGS{9812370,
           author={Paul, Steve and Ghassemi, Payam and Chowdhury, Souma},
           booktitle={2022 International Conference on Robotics and Automation (ICRA)}, 
           title={Learning Scalable Policies over Graphs for Multi-Robot Task Allocation using Capsule Attention Networks}, 
           year={2022},
           volume={},
           number={},
           pages={8815-8822},
           doi={10.1109/ICRA46639.2022.9812370}}


If you have any questions or concerns, please raise an issue or email: stevepau@buffalo.edu
                
                
