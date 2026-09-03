#!/usr/bin/env bash

script_path=$(dirname "$(realpath "$0")")

# project name
project_string=$(grep project_name= "${script_path}/project.conf" | grep -v [#] )
project_name=$(cut -d'"' -f2 <<<"$project_string")


echo "Looking for docker:" $project_name
echo "Currently the following containers are running. Note, the script selects the first match."

IFS=$'\n'
count=1
for docker_name in $(docker ps | grep -v STATUS)
do
 # echo $docker_name
 echo "${count})" $(echo $docker_name | cut -d \"  -f1) 
 count=$((count+1))
done
unset IFS


docker_id=$(docker ps | grep -F "$project_name" | cut -d$'\n' -f1 | cut -d' ' -f1)

if [ -z "$docker_id" ]; then
	echo "No running container found matching project: $project_name"
	exit 1
fi

echo "Connecting to docker: $docker_id"

docker exec -it "$docker_id" /bin/bash
