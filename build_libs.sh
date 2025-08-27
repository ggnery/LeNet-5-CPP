SOURCE_DIR=$(pwd)
INSTALL_DIR=$SOURCE_DIR/install

set -e # Exit on error

rm -rf libs build
mkdir -p libs && cd libs

# torch
wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip
unzip libtorch-shared-with-deps-2.8.0+cpu.zip

rm -rf libtorch-shared-with-deps-2.8.0+cpu.zip