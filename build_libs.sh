SOURCE_DIR=$(pwd)
INSTALL_DIR=$SOURCE_DIR/install

set -e # Exit on error

rm -rf libs install build
mkdir -p libs && cd libs

# Build dlib
wget http://dlib.net/files/dlib-20.0.tar.bz2
tar -xjvf dlib-20.0.tar.bz2 && rm dlib-20.0.tar.bz2

cd dlib-20.0
mkdir -p build && cd build

cmake .. -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR/dlib
cmake --build . --config Release
cmake --install .