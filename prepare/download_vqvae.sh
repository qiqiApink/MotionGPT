mkdir -p checkpoints
cd checkpoints

echo "The pretrained_vqvae will be stored in the './checkpoints' folder"
echo "Downloading"
gdown "https://drive.google.com/uc?id=1A4cfdodZbiENV75tR9IErei9yiZGlosT"

echo "Extracting"
unzip pretrained_vqvae.zip

echo "Cleaning"
rm pretrained_vqvae.zip

echo "Downloading done!"
