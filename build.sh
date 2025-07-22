#!/bin/bash
echo "🔧 Installing Git LFS..."
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs -y

echo "🔧 Running git lfs pull..."
git lfs install
git lfs pull

# Check if the file downloaded correctly
echo "🔍 Model file size:"
ls -lh knn_model.pkl
