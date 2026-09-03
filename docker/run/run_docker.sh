#!/usr/bin/env bash

CONTAINER_NAME=""
CONTAINER_NAME_ARGS=()
DETACH=false

is_rootless_docker() {
	if ! command -v docker >/dev/null 2>&1; then
		echo "docker command is required but was not found in PATH." >&2
		return 2
	fi

	local rootless_flag
	if rootless_flag=$(docker info --format '{{.Rootless}}' 2>/dev/null); then
		if [[ "${rootless_flag,,}" == "true" ]]; then
			return 0
		fi
		return 1
	fi

	if docker info 2>/dev/null | grep -qi "rootless"; then
		return 0
	fi

	return 1
}

# Argument parsing loop
while [[ $# -gt 0 ]]
do
    key="$1"
    case $key in
        --name)
        CONTAINER_NAME="$2"
        CONTAINER_NAME_ARGS=(--name "$2")
        shift # Remove --name from processing
        shift # Remove the container name from processing
        ;;
        -d|--detach)
        DETACH=true
        shift
        ;;
        *)
        echo "Skipping unknown option: $1"
        shift # Remove the unknown option
        ;;
    esac
done

if [[ -n "$CONTAINER_NAME" ]]; then
    echo "CONTAINER_NAME set to: $CONTAINER_NAME"
else
    echo "CONTAINER_NAME set to: (not set)"
fi

# Do we have nvidia 
nvidia_exist=$(command -v nvidia-smi)
if [ -z "$nvidia_exist" ]; then
	echo "nvidia-smi not found. Please install the NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
	exit 1
else
	echo "NVIDIA driver found."
fi


# get the project name and the image name from the config file in the root directory
script_path=$(dirname "$(realpath "$0")")

# project name
project_string=$(grep project_name= "${script_path}/../../project.conf" | grep -v [#] )
project_name=$(cut -d'"' -f2 <<<"$project_string")

# base image (full reference including namespace); strip namespace for tagging
image_string=$(grep base_image= "${script_path}/../../project.conf" | grep -v [#])
base_image=$(cut -d'"' -f2 <<<"$image_string")
image_name="${base_image##*/}"

# project workspace
workspace_string=$(grep project_workspace= "${script_path}/../../project.conf" | grep -v [#] )
WORKSPACE=$(cut -d'"' -f2 <<<"$workspace_string")

# project data container
data_container_string=$(grep project_data_container= "${script_path}/../../project.conf" | grep -v [#] )
DATA_CONTAINER=$(cut -d'"' -f2 <<<"$data_container_string")

# shared memory size
shm_size_string=$(grep shm_size= "${script_path}/../../project.conf" | grep -v [#] )
SHM_SIZE=$(cut -d'"' -f2 <<<"$shm_size_string")

image_name=${image_name}_${project_name}_${USER} 
echo "Running image [ ${image_name} ]"
echo "In workspace [ ${WORKSPACE} ]"
echo "Data container [ ${DATA_CONTAINER} ]"
echo "Shared memory size [ ${SHM_SIZE} ]"

is_rootless_docker
rootless_status=$?
if [[ $rootless_status -eq 2 ]]; then
	exit 1
fi

ROOTLESS_DOCKER=false
if [[ $rootless_status -eq 0 ]]; then
	ROOTLESS_DOCKER=true
	echo "Rootless Docker detected. Skipping container user overrides."
else
	echo "Root Docker daemon detected. Enabling container user overrides."
fi

CONTAINER_HOME="/home/$USER"
if [[ "$ROOTLESS_DOCKER" == "true" ]]; then
	CONTAINER_HOME="/home/$USER/$WORKSPACE"
fi

if [[ "$DETACH" == "true" ]]; then
	docker_cmd=(docker run -d)
else
	docker_cmd=(docker run -it)
fi

if [[ ${#CONTAINER_NAME_ARGS[@]} -gt 0 ]]; then
	docker_cmd+=("${CONTAINER_NAME_ARGS[@]}")
fi

docker_cmd+=(
	--runtime=nvidia
	--network=host
	--ipc=host
)

if [[ "$ROOTLESS_DOCKER" == "false" ]]; then
	docker_cmd+=(--privileged --user="$USER" --group-add plugdev --group-add video --group-add dialout)
fi

docker_cmd+=(
	--env "USER=$USER"
	--env "HOME=${CONTAINER_HOME}"
	--env "DISPLAY=${DISPLAY}"
	--env "XAUTHORITY=/tmp/.Xauthority.host"
	--env "XDG_RUNTIME_DIR=/tmp"
	--shm-size="${SHM_SIZE}"
	--env HOME="${CONTAINER_HOME}"
	--workdir="${CONTAINER_HOME}"
	--volume="/home/$USER/$WORKSPACE:${CONTAINER_HOME}"
	--volume="$DATA_CONTAINER:/data:rw"
	# act repo (code + 76G untracked data/, both needed together since
	# imitate_episodes.py resolves data/results paths relative to cwd) lives
	# on the host at /home/$USER/code/act, not inside this workspace repo.
	# Bind-mounted live instead of cloned/copied: zero-copy, stays in sync
	# with host edits/git ops, and checkpoints written to results/ land
	# directly on host disk.
	--volume="/home/$USER/code/act:${CONTAINER_HOME}/act:rw"
	--volume="/etc/timezone:/etc/timezone:ro"
	--volume="/etc/localtime:/etc/localtime:ro"
	--volume="/home/$USER/.ssh:/home/$USER/.ssh:rw"
	--volume="/tmp/.X11-unix:/tmp/.X11-unix:rw"
	--volume="/dev:/dev"
        --volume="${XAUTHORITY:-$HOME/.Xauthority}:/tmp/.Xauthority.host:ro"

)

docker_cmd+=(
	--env QT_X11_NO_MITSHM=1
	--env NO_AT_BRIDGE=1
)

if [[ "$ROOTLESS_DOCKER" == "false" ]]; then
	docker_cmd+=(
		--volume="/etc/group:/etc/group:ro"
		--volume="/etc/passwd:/etc/passwd:ro"
		--volume="/etc/shadow:/etc/shadow:ro"
		--volume="/etc/sudoers.d:/etc/sudoers.d:ro"
	)
fi

if [[ "$DETACH" == "true" ]]; then
	# Detached/headless: run the entrypoint then stay alive in the background.
	# Attach later with ./exec-docker.sh. Auto-removed on stop (--rm).
	docker_cmd+=(--rm --entrypoint "${CONTAINER_HOME}/docker/run/entrypoint.sh" "$image_name" tail -f /dev/null)
else
	docker_cmd+=(--rm --entrypoint "${CONTAINER_HOME}/docker/run/entrypoint.sh" "$image_name" bash)
fi

echo "Executing: ${docker_cmd[*]}"
"${docker_cmd[@]}"
