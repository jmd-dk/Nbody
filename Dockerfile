# Base
FROM debian:10.7-slim
SHELL ["/usr/bin/env", "bash", "-c"]
CMD ["bash"]

# Installation options
ARG concept_version=/source
ARG top_dir="/concept"
ARG slim=True
ARG mpi=mpich
ARG mpi_configure_options="+= --with-device=ch3:sock"

# Build
COPY \
    .env* \
    .gitignore* \
    COPYING* \
    Dockerfile* \
    README.md* \
    installer \
    /source/
COPY concept* /source/concept/
COPY .github* /source/.github/
ARG DEBIAN_FRONTEND=noninteractive
RUN : \
    # Update APT cache
    && apt-get update \
    && apt-get install -y --no-install-recommends apt-utils \
        2> >(grep -v 'apt-utils is not installed' >&2) \
    # Install CO𝘕CEPT
    && bash /source/installer -y "${top_dir}" \
    && rm -rf /source \
    # Set up CO𝘕CEPT and Python environment
    && sed -i "1i source \"${top_dir}/concept/concept\"" ~/.bashrc \
    && apt-get install -y --no-install-recommends less \
    && echo "[ ! -t 0 ] || alias less='less -r -f'" >> ~/.bashrc \
    && ln -s "${top_dir}/python/bin/python3" "${top_dir}/python/bin/python" \
    # Set up Bash autocompletion
    && apt-get install -y --no-install-recommends bash-completion \
    && echo "[ ! -t 0 ] || source /etc/bash_completion" >> ~/.bashrc \
    # Set up Bash history search with ↑↓
    && echo "[ ! -t 0 ] || bind '\"\e[A\": history-search-backward'" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || bind '\"\e[B\": history-search-forward'" >> ~/.bashrc \
    # Set up color prompt
    && echo "[ ! -t 0 ] || PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\\$ '" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || alias ls='ls --color=auto'" >> ~/.bashrc \
    && echo "[ ! -t 0 ] || alias grep='grep --color=auto'" >> ~/.bashrc \
    # Remove unnecessary packages and clean APT cache
    && apt-get autoremove -y \
    && apt-get purge -y 'g++*' 'libstdc++*dev' 'gfortran*' 'libgfortran*dev' \
    && apt-get clean -y \
    && apt-get autoclean -y \
    && rm -rf /var/lib/{apt/lists,cache,log}/* \
    && find /var/lib/dpkg/info/* -not -name '*.list' -delete \
    # Remove other caches
    && rm -rf /tmp/* ~/.cache/*
ENV \
    PATH="${PATH}:${top_dir}/concept:${top_dir}/python/bin" \
    TERM="linux"
WORKDIR "${top_dir}/concept"

