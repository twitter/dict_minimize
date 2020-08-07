#!/bin/bash

set -ex
set -o pipefail

# Display what version is being used for logging
python3 --version

# Fail if untracked files so we don't delete them in next step
test -z "$(git status --porcelain)"

# Build from clean repo, delete all ignored files
git clean -x -ff -d

# Log the git version inside of the wheel file
SHA_LONG=$(git rev-parse HEAD)
echo VERSION=\"$SHA_LONG\" >version.log

# Now the actual build
python3 setup.py sdist
