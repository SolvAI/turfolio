#!/bin/sh
# Turfolio installer for macOS and Linux:  curl -LsSf https://raw.githubusercontent.com/SolvAI/turfolio/main/install.sh | sh
# (or ./install.sh inside a clone). Installs uv if needed, Python 3.12 + dependencies in ./.venv, then runs the one-time setup.
set -e
REPO="https://github.com/SolvAI/turfolio"
say() { printf '\033[36;1m>\033[0m %s\n' "$1"; }

# 1. the code (skip when run from inside a clone)
if [ ! -f turfolio.py ]; then
  if [ -f "$(dirname "$0")/turfolio.py" ]; then
    cd "$(dirname "$0")"
  else
    say "Downloading Turfolio into ./turfolio"
    if command -v git >/dev/null 2>&1; then
      git clone --depth 1 "$REPO.git" turfolio
    else
      curl -LsSf "$REPO/archive/refs/heads/main.tar.gz" | tar xz && mv turfolio-main turfolio
    fi
    cd turfolio
  fi
fi

# 2. uv (fast Python package manager; brings its own Python)
if ! command -v uv >/dev/null 2>&1; then
  say "Installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

# 3. Python 3.12 + dependencies (CPU build of PyTorch on Linux: 200 MB instead of 2.5 GB of CUDA libraries)
say "Installing Python 3.12 and the dependencies into ./.venv"
uv venv --quiet --allow-existing --python 3.12 .venv
if [ "$(uname -s)" = "Linux" ]; then
  uv pip install --quiet --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
fi
uv pip install --quiet --python .venv/bin/python -r requirements.txt
chmod +x turfolio

# 4. one-time data setup
./turfolio setup
printf '\n\033[1mInstalled.\033[0m  cd %s, then:  ./turfolio bets --watch\n' "$(pwd)"
