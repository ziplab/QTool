
# check current task, the script would run the next round experiment when current pid is finished
pid=$1
config=$2
repeat=1

if [ "$3" != "" ]; then repeat=$3; fi

script=train.sh

if [ "$pid" != "" ]
then
  while true
  do
    nvidia-smi | grep $pid
    if [ $? -eq 0 ]; then sleep 1m; continue; else break; fi
  done
  #sleep 10

  while [ $repeat -gt 0 ]
  do
    echo "starting script"
    bash $script $config
    if [ $? -eq 0 ]; then break; else sleep 1m; fi
    repeat=$(( $repeat - 1 ))
  done
fi

