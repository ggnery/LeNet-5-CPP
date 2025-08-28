SOURCE_DIR=$(pwd)
INSTALL_DIR=$SOURCE_DIR/install

set -e # Exit on error

# Function to display menu and get user choice
choose_libtorch_version() {
    echo "Choose LibTorch version:"
    echo "1) CPU only"
    echo "2) CUDA 12.9"
    echo "3) CUDA 12.8"
    echo "4) CUDA 12.6"
    echo -n "Enter your choice (1-4): "
    read choice
    
    case $choice in
        1)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip"
            LIBTORCH_FILE="libtorch-shared-with-deps-2.8.0+cpu.zip"
            echo "Selected: CPU version"
            ;;
        2)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu129/libtorch-shared-with-deps-2.8.0%2Bcu129.zip"
            LIBTORCH_FILE="libtorch-shared-with-deps-2.8.0+cu129.zip"
            echo "Selected: CUDA 12.9 version"
            ;;
        3)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu128/libtorch-shared-with-deps-2.8.0%2Bcu128.zip"
            LIBTORCH_FILE="libtorch-shared-with-deps-2.8.0+cu128.zip"
            echo "Selected: CUDA 12.8 version"
            ;;
        4)
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cu126/libtorch-shared-with-deps-2.8.0%2Bcu126.zip"
            LIBTORCH_FILE="libtorch-shared-with-deps-2.8.0+cu126.zip"
            echo "Selected: CUDA 12.6 version"
            ;;
        *)
            echo "Invalid choice. Defaulting to CPU version."
            LIBTORCH_URL="https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.8.0%2Bcpu.zip"
            LIBTORCH_FILE="libtorch-shared-with-deps-2.8.0+cpu.zip"
            ;;
    esac
}

# Get user's choice
choose_libtorch_version

rm -rf libs build
mkdir -p libs && cd libs

# Download selected LibTorch version
echo "Downloading LibTorch..."
if wget $LIBTORCH_URL; then
    echo "Download completed successfully."
    echo "Extracting LibTorch..."
    unzip $LIBTORCH_FILE
    echo "Extraction completed."
    # Clean up downloaded zip file
    rm -rf $LIBTORCH_FILE
    echo "Cleaned up zip file."
else
    echo "Error: Failed to download LibTorch. Please check your internet connection and try again."
    exit 1
fi