
if [ -e .env ]; then
  source .env
  if [ "$3" == "load_data" ]; then
    copy_imagenet_to_ddr
  fi
fi

if [ "$FASTDIR" == "" ]; then
  FASTDIR=/workspace
fi

if [ -d $FASTDIR/git/aim-uofa-model-quantization ]; then
  cd $FASTDIR/git/aim-uofa-model-quantization
elif [ -d /workspace/git/aim-uofa-model-quantization ]; then
  cd /workspace/git/aim-uofa-model-quantization
else
  FASTDIR=../..
  cd .
fi


script=main.py
train_batch=20
val_batch=20
dataset='imagenet'
root=$FASTDIR/data/imagenet
case='fake'
keybase=''
keyword=','
model='unknow'
base=1
epochs=0
options=''
pretrain='none'

config=config.bin
if [ "$1" != "" ]; then config=$1; fi
if [ -e $config ];
then
  echo "Loading config from $config"
  source $config
fi

if [ "$DELAY" != "" ]; then
  delay=$DELAY
else
  delay=0
fi

if [ "$2" != "" ]; then script=$2; fi

nvidia-smi

python $script --dataset $dataset --root $root \
  --model $model --base $base \
  --epochs $epochs -b $train_batch -v $val_batch \
  --case $case --keyword $keyword \
  --delay $delay \
  $options

result=$?
echo "python result $result"
if [ "$result" -ne "0" ];
then
  echo $PATH
  echo $LD_LIBRARY_PATH
  which python
  python -V
fi

cd -

#notify-send "cmd finished in $0" "`date`"


