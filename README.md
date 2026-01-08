# BitDistill

### TODO

* Fix the training speed problem when using NVIDIA GPUs. (**We run all experiments on AMD Mi300x**)
* Support more types of models (now only Qwen series are supported).

### Dataset

Please organize the dataset in the format used by [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main/data).

### Docker

* AMD: `rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.5.1`
* NVIDIA: `yushuiwx/rl:v2.0.2`

### Environments setup

* AMD

```
bash mi300_setup.sh
```

* NVIDIA

```
bash setup.sh
```

### Training Commands

qwen3 series exps please refer to 
```
bash qwen3-exp.sh
```


* for training deepseekdistill fp16 baseline on downstream task:
    * $lr: learning rate
    * $model: Qwen model name
    * $gpu: gpu index for training
```
bash ds-exp-run-sft-baseline.sh $lr $model $gpu
```

* for training deepseekdistill bitdistill on downstream task using fp16 baseline as teacher:

    * $teacher: local path for fp16 huggingface-format teacher model
    * $beta: loss weight for logits distillation
    * $minilmweight: loss weight for minilm v2 distillation
    * $distilllayer: use which layer to apply minilm v2 distillation
```
bash ds-exp-run-sft-bitdistill.sh $lr $model $gpu $teacher $beta $minilmweight $distilllayer  
```

### BitNet Model Test Demo

```
./test-ds-model/test.sh
```
