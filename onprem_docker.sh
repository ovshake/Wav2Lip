
# Script to create container using given docker image for vwm.

# Example command to run:
# $ bash onprem_create_container.sh -g 0 -n xyz -p 8001

# Note: `sh onprem_create_container.sh ...` shall not work.

# Args:
# -g: GPU number, for a non-GPU machine, pass -1,
#     to use all GPUs, pass -g all or do not pass at all,
#     If you pass a single GPU, then note that the GPU ID in the container will be 0
# -n: name of the container
# -p: port number (this is needed if you want to start jupyter lab on a remote machine)

image=nvcr.io/nvidia/pytorch:22.03-py3
anthro_gid=3001

# get inputs
while getopts "g:n:p:i:" OPTION; do
    case $OPTION in
        g) gpu=$OPTARG;;
        n) name=$OPTARG;;
        p) port=$OPTARG;;
        i) image=$OPTARG;;
        *) exit 1 ;;
    esac
done


# nvidia-docker command works only for GPU machines
command="docker"

container_name=gpu-"$gpu"_"$name"_"$port"
echo "=> Firing docker container $container_name from $image with $command"

# if -g is not passed, use all GPUs
if [ -z "$gpu" ]
  then
    echo "Using all GPUs"
    gpu="all"
fi

$command run --rm -it \
    --name $container_name \
    --user=$(id -u):$anthro_gid \
    --gpus device=$gpu \
    -p $port:$port \
    -v /home/users/abhishekm/:/home/users/abhishekm/ \
    -v /home/users/$USER/.torch/:/home/users/$USER/.torch/ \
    -v /home/users/$USER/.cache/:/home/users/$USER/.cache/ \
    -v /home/users/$USER/.bash_history:/home/users/$USER/.bash_history \
    -v /home/users/$USER/.local:/.local \
    --env IMAGE_NAME=$image \
    --env JUPYTER_DATA_DIR=/.local/share/jupyter \
    --env JUPYTER_RUNTIME_DIR=/.local/share/jupyter/runtime \
    --env JUPYTER_CONFIG_DIR=/.jupyter \
    --env IPYTHONDIR=/.ipython \
    --hostname $HOSTNAME \
    --ipc host \
    --shm-size 16G \
    $image /bin/bash
