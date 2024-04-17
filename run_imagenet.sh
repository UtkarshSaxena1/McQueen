#!/bin/sh
source setup.sh
echo "Log directory: Run1"
logdir=$1
logger="./logger/resnet18_Run1"
mkdir ${logger}
log_file="${logger}/Run1.txt"
codedir="./examples/classifier_imagenet/"
cd "${codedir}/prototxt/"
sed -i "/log_name:/c\log_name: 'Run1'" resnet_multigpu.prototxt 
cd -
python examples/classifier_imagenet/main.py --hp ./examples/classifier_imagenet/prototxt/resnet_multigpu.prototxt

