mkdir -p checkpoints
cd checkpoints

echo "The evaluators will be stored in the './checkpoints' folder"
echo "Downloading"
gdown "https://drive.google.com/uc?id=1jD08gNAU2zVKDAMVyRbxzzv2uA9ssqMk"
gdown "https://drive.google.com/uc?id=1caLMTO5EMZoaCY2U7yEgZp3dG1seyNKF"

echo "Extracting"
unzip t2m.zip
unzip kit.zip

echo "Cleaning"
rm t2m.zip
rm kit.zip

echo "Downloading done!"
