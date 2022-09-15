pip install faiss-gpu
pip install tensorboardx
cd ..
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir ./
cd ..
mkdir data
cd data
cp /content/drive/MyDrive/Antibody/Data/Antigens/antigens_split_processed_windowed_250_125.tar.gz .
tar -xzvf antigens_split_processed_windowed_250_125.tar.gz
cp /content/drive/MyDrive/Antibody/Data/OAS/Cleaned/cleaned_antibodies.tar.gz .
tar -xzvf cleaned_antibodies.tar.gz
cp /content/drive/MyDrive/Antibody/Data/Parallel/*.pth .
cd ../AntiXLM
