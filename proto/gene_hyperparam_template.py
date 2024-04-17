import os

import google.protobuf as pb
import google.protobuf.text_format

from proto import efficient_pytorch_pb2 as eppb


def main():
    gene_base_template()


def gene_base_template():
    root_dir = os.getenv('CURRENT_DIR')
    # default values
    hyper = eppb.HyperParam()
    hyper.main_file = hyper.main_file
    hyper.arch = hyper.arch
    hyper.model_source = hyper.model_source
    hyper.debug = hyper.debug
    hyper.overfit_test = hyper.overfit_test
    hyper.log_name = hyper.log_name
    hyper.data = '/home/{}/datasets/data.cifar10'.format(os.getenv('USER'))
    hyper.workers = hyper.workers
    hyper.batch_size = hyper.batch_size
    hyper.print_freq = hyper.print_freq
    hyper.evaluate = hyper.evaluate
    hyper.pretrained = hyper.pretrained
    hyper.lr = hyper.lr
    hyper.lr_stage1 = hyper.lr_stage1
    hyper.lr_stage2 = hyper.lr_stage2
    hyper.lr_stage3 = hyper.lr_stage3
    hyper.lr_stage4 = hyper.lr_stage4
    hyper.epochs = hyper.epochs
    hyper.epoch_stage1 = hyper.epoch_stage1
    hyper.epoch_stage2 = hyper.epoch_stage2
    hyper.epoch_stage3 = hyper.epoch_stage3
    hyper.epoch_stage4 = hyper.epoch_stage4
    hyper.start_epoch = hyper.start_epoch
    hyper.ensemble = hyper.ensemble
    hyper.resume_ensemble = hyper.resume_ensemble
    
    hyper.resume = hyper.resume
    hyper.weight = hyper.weight

    hyper.target_bops = hyper.target_bops
    hyper.gamma_bop = hyper.gamma_bop

    hyper.seed = hyper.seed
    hyper.export_onnx = hyper.export_onnx

    hyper.gpu_id = hyper.gpu_id
    hyper.multi_gpu.world_size = hyper.multi_gpu.world_size
    hyper.multi_gpu.rank = hyper.multi_gpu.rank
    hyper.multi_gpu.dist_url = hyper.multi_gpu.dist_url
    hyper.multi_gpu.dist_backend = hyper.multi_gpu.dist_backend
    hyper.multi_gpu.multiprocessing_distributed = hyper.multi_gpu.multiprocessing_distributed

    hyper.qmode = hyper.qmode
    hyper.nbits_w = hyper.nbits_w
    hyper.nbits_a = hyper.nbits_a

    hyper.warmup.epochs = hyper.warmup.epochs
    hyper.warmup.multiplier = hyper.warmup.multiplier
    # print(hyper.HasField("warmup"))

    hyper.optimizer = hyper.optimizer
    hyper.sgd.weight_decay = hyper.sgd.weight_decay
    hyper.sgd.momentum = hyper.sgd.momentum
    hyper.adam.weight_decay = hyper.adam.weight_decay

    hyper.lr_scheduler = hyper.lr_scheduler
    # StepLR
    hyper.step_lr.step_size = hyper.step_lr.step_size
    hyper.step_lr.gamma = hyper.step_lr.gamma
    # MultiStepLR
    hyper.multi_step_lr.milestones.extend([20, 30, 50])
    # list(hyper.multi_step_lr.milestones) convert it to list
    hyper.multi_step_lr.gamma = hyper.multi_step_lr.gamma
    # CyclicLR
    hyper.cyclic_lr.base_lr = hyper.cyclic_lr.base_lr
    hyper.cyclic_lr.max_lr = hyper.cyclic_lr.max_lr
    hyper.cyclic_lr.step_size_up = hyper.cyclic_lr.step_size_up
    hyper.cyclic_lr.mode = hyper.cyclic_lr.mode
    hyper.cyclic_lr.gamma = hyper.cyclic_lr.gamma
    with open(os.path.join(root_dir, 'proto', 'template.prototxt'), 'w') as wf:
        print(hyper)
        print('Writing hyper parameter at {}/proto/template.prototxt'.format(root_dir))
        wf.write(str(hyper))


"""
Load hyperparameter from prototxt.
```python
person2 = addressbook_pb2.Person()
with open('person.prototxt', 'r') as rf:
    pb.text_format.Merge(rf.read(), person2)
```
"""


def dump_object(obj):
    for descriptor in obj.DESCRIPTOR.fields:
        value = getattr(obj, descriptor.name)
        if descriptor.type == descriptor.TYPE_MESSAGE:
            if descriptor.label == descriptor.LABEL_REPEATED:
                map(dump_object, value)
            else:
                dump_object(value)
        else:
            print("%s: %s" % (descriptor.full_name, value))


if __name__ == '__main__':
    main()
