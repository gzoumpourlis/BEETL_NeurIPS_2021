wget https://figshare.com/ndownloader/files/30701264 -O ./BeetlMI_test.zip
unzip ./BeetlMI_test.zip
mkdir ~/mne_data/MNE-beetlmitest-data
cp -r ./finalMI/* ~/mne_data/MNE-beetlmitest-data
rm -r ./finalMI
rm ./BeetlMI_test.zip
