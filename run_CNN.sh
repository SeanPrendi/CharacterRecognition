# Delete existing screen
for session in $(sudo screen -ls | grep -o '[0-9]\{5\}.character_recognition')
do
sudo screen -S "${session}" -X quit
done

# Delete the log
sudo rm screenlog.0

# For python in conda env
sudo screen -dmSL "character_recognition" bash -c "export PATH='/opt/miniconda3/bin:$PATH'; . '/opt/miniconda3/etc/profile.d/conda.sh'; conda activate CharacterRecognition; python CNN.py"

# List screens
sudo screen -ls
