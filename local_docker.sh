docker run --privileged --net=host --ipc=host --gpus=all -v $(pwd):$(pwd) -w $(pwd) -it yushuiwx/rl:v2.0.2


sudo docker run -it --privileged --net=host --ipc=host --rm -v $(pwd):$(pwd) -w $(pwd) rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.5.1 bash
