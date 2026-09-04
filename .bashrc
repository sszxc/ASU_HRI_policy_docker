# ~/.bashrc: executed by bash(1) for non-login shells.

# --- asu_il_policy_ws: ROS 2 / Fast DDS -------------------------------------
# These MUST stay ABOVE the non-interactive guard below, so scripts and
# `docker exec` shells inherit them and not just interactive logins.
#
# Copied verbatim from asu_state_reward_ws (same host = univ.brain.left,
# 192.168.1.26 -- see fastdds_profile.xml, which pins the lab NIC and lists
# peers explicitly because the managed switch's IGMP snooping prunes DDS
# discovery multicast). That XML is the single source of truth for peers --
# do NOT add ROS_STATIC_PEERS or any other FASTRTPS_* variable anywhere in
# this file. Needed for ros2_ws/src/policy_runner to see the camera/joint
# topics published on univ.perception (192.168.1.25).
export ROS_DOMAIN_ID=10
export ROS_AUTOMATIC_DISCOVERY_RANGE=SUBNET
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/fastdds_profile.xml

# Source ONLY jazzy (the image also ships noetic + ros1_bridge; mixing them
# in one shell breaks colcon).
source /opt/ros/jazzy/setup.bash
if [ -f "$HOME/ros2_ws/install/setup.bash" ]; then
    source "$HOME/ros2_ws/install/setup.bash"
fi

# `pip3 install --user` (no --break-system-packages, PEP668 fallback) drops
# console scripts here -- e.g. the `rerun` viewer binary rerun-sdk ships,
# needed by act_infer_mujoco.py's `--rerun`. Above the guard so `docker exec`
# shells get it too, same reasoning as the ROS vars above.
export PATH="$HOME/.local/bin:$PATH"

# If not running interactively, don't do anything else
case $- in
    *i*) ;;
      *) return;;
esac

HISTCONTROL=ignoreboth
shopt -s histappend
HISTSIZE=1000
HISTFILESIZE=2000
shopt -s checkwinsize

[ -x /usr/bin/lesspipe ] && eval "$(SHELL=/bin/sh lesspipe)"

if [ -z "${debian_chroot:-}" ] && [ -r /etc/debian_chroot ]; then
    debian_chroot=$(cat /etc/debian_chroot)
fi

force_color_prompt=yes
if [ -n "$force_color_prompt" ]; then
    if [ -x /usr/bin/tput ] && tput setaf 1 >&/dev/null; then
        color_prompt=yes
    fi
fi
if [ "$color_prompt" = yes ]; then
    PS1='${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@docker\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ '
else
    PS1='${debian_chroot:+($debian_chroot)}\u@\h:\w\$ '
fi
unset color_prompt force_color_prompt

if [ -x /usr/bin/dircolors ]; then
    test -r ~/.dircolors && eval "$(dircolors -b ~/.dircolors)" || eval "$(dircolors -b)"
    alias ls='ls --color=auto'
    alias grep='grep --color=auto'
fi

alias ll='ls -alF'
alias la='ls -A'
alias l='ls -CF'

if [ -f ~/.bash_aliases ]; then
    . ~/.bash_aliases
fi

if ! shopt -oq posix; then
  if [ -f /usr/share/bash-completion/bash_completion ]; then
    . /usr/share/bash-completion/bash_completion
  elif [ -f /etc/bash_completion ]; then
    . /etc/bash_completion
  fi
fi

alias colcon_build='colcon build --symlink-install --packages-select policy_runner  && source install/setup.bash'
