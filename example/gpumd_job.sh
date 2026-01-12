#!/bin/bash
#SBATCH --job-name=gpumd   # 作业名称
#SBATCH --nodes=1                       # 请求一个节点
#SBATCH --ntasks=1                      # 请求一个进程（ASE 是单进程运行）
#SBATCH --partition=debug                 # 请求 GPU 分区（根据集群配置修改）
#SBATCH --gres=gpu:1                    # 请求 1 个 GPU
#SBATCH --time=7-00:00:00
#SBATCH --output=job-%j.out                 # 标准输出文件（使用作业ID作为文件名）
#SBATCH --error=job-%j.err                  # 错误输出文件（使用作业ID作为文件名）

#conda run -n ase_env --no-capture-output python -u ase_gpumd.py
gpumd
#python -u pysed.py
