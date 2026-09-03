#!/bin/bash
# Runs once when the container starts, before `exec "$@"` replaces this
# script with the passed command. `docker exec` (exec-docker.sh) bypasses
# this entirely — put per-shell setup in ~/.bashrc instead.
set -e

##### User editable. Non-zero exit here kills container startup.

# `detr` (act/detr/) isn't baked into the image -- the act repo is bind-mounted
# live at $HOME/act (see run_docker.sh), so its editable install has to be
# redone on every fresh container. --no-deps: detr declares no dependencies of
# its own, and this avoids pip re-touching the pinned requirements.txt stack.
if [ -d "$HOME/act/detr" ]; then
    pip3 install -e "$HOME/act/detr" --no-deps --break-system-packages -q
fi

##### End of user editable

exec "$@"
