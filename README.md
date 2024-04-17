# McQueen 
This repository is the official implementation of the paper : [McQueen : Mixed Precision Quantization of Early Exit Networks](https://papers.bmvc2023.org/0511.pdf) which appeared in BMVC 2023.
## Versions
Pytorch version: 1.10.2 <br />
CUDA Version: 12.0 <br />
check environment.yml for more info. <br />

## Important files
The repository provides codebase for obtaining results presented in the paper on ImageNet with ResNet-18 model. 
Important files: 
1. python main file (./examples/classifier_imagenet/main.py)
2. Model description file (./models/imagenet/)
    resnet.py containes the multi exit resnet-18 architecture 
    ensemble.py provides implementation of ensemble model. 
3. Quantizer (./models/_modules/lsq.py)
4. train function which implements gradient masking (./examples/__init__.py)
5. Hyperparameter file (./examples/classifier_imagenet/prototxt/resnet_multigpu.prototxt)
6. Bash file to run (./run_imagenet.sh)

## Hyperparameters
- arch: Model architecture (currently only supports resnet18)
- log_name: Name of the training run
- lr_stageX: stagewise learning rate assignment
- epoch_stageX: stagewise epoch assignment
- batch_size: Training and validation batchsize
- workers: Number of workers in dataloader
- print_freq: Print training status every print_freq iterations
- evaluate: Flag if only validating the model and not training. (set false if training)
- ensemble: Flag if ensemble model needs to be used.
- resume_ensemble: if evaluating, provide location of ensemble model here
- resume: provide location if resuming from a checkpoint.
- pretrained: set to true if using pytorch pretrained model as backbone
- start_epoch : if resuming from a checkpoint, provide the epoch to start from.
- data: locating of imagenet dataset
- nbits_w: Weight precision initialized
- nbits_a: Activation precision initialized
- target_bops : Target Bitwise Operations
- gamma_bop : BOPs regularization weight

## Running the code
1. Setup the environment
```
conda env create -f environment.yml
```
2. Add location of imagenet data directory in ./examples/classifier_imagenet/prototxt/resnet_multigpu.prototxt
3. Run the code
```
run_imagenet.sh
```
## References
If you use this code, please cite the following paper:
```
@article{saxena2023mcqueen,
  title={McQueen: Mixed Precision Quantization of Early Exit Networks},
  author={Saxena, Utkarsh and Roy, Kaushik},
  booktitle = {British Machine Vision Conference, BMVC 2023},
  year={2023}
}
```
```
