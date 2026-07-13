# manually download hm3d and objectnav

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
REPO_PATH="$(cd "$SCRIPT_DIR" && git rev-parse --show-toplevel 2>/dev/null)"
if [ -z "$REPO_PATH" ]; then
  echo "Unable to determine git repository root from $SCRIPT_DIR" >&2
  exit 1
fi
source "$SCRIPT_DIR/utils/input_utils.sh"
source "$SCRIPT_DIR/utils/output_utils.sh"
source "$SCRIPT_DIR/utils/pkg_utils.sh"

prompt="Detected Repo Path as '${REPO_PATH}'. Confirm [Y/n]?  "
yes_action=( : )  # no-op for yes
no_action=( get_str_input "Please enter the correct repo path: " REPO_PATH )

get_yn_input "$prompt" Y yes_action[@] no_action[@]

VLN_Folder=$REPO_PATH
print_ok "Repo Folder set to: '$VLN_Folder'"

DATA_Folder=$VLN_DATA_DIR
if [ -z $DATA_Folder ]; then
  get_str_input "Unable to Detect Data Folder. Where is it? " DATA_Folder
else
  yes_action=( : )  # no-op for yes
  no_action=( get_str_input "Please enter the correct data folder path: " DATA_Folder )
  get_yn_input "Detected Data Folder as '$DATA_Folder'. Is this correct [Y/n]? " Y yes_action[@] no_action[@]
fi

while [ ! -d $DATA_Folder ]; do
  yes_action=( mkdir -p "$DATA_Folder" )
  no_action=( get_str_input "Please enter the correct data folder path: " DATA_Folder )
  get_yn_input "Folder: '$DATA_Folder' does not exist. Would you like to create it? [Y/n] " Y yes_action[@] no_action[@]
done

print_ok "Data Folder set to: '$DATA_Folder'"

read -rp "Download datasets? [Y/n]  " answer
answer=${answer:-Y}

case "$answer" in
    [Yy]* ) 
        echo "Follow instructions at this link: https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#downloading-hm3d-with-the-download-utility"
        read -p "Token Secret:  " token_secret
        read -p "Token ID:  " token_id
        (cd $DATA_Folder; python3 -m habitat_sim.utils.datasets_download --username $token_id --password $token_secret --data-path $DATA_Folder --uids hm3d_val_habitat)
        ;;
    [Nn]* ) 
        echo "Assuming datasets are already downloaded. Script will fail if this is false."
        ;;
    * ) 
        echo "Invalid response (please enter y or n)  "
        echo "Exiting..."
        exit 1
        ;;
esac

cd $VLN_Folder
if [ -d "data" ]; then
    rm -rf data
fi
mkdir -p data
mkdir -p data/scene_datasets
ln -s $DATA_Folder/versioned_data data/versioned_data
ln -s $DATA_Folder/versioned_data/hm3d-0.2/hm3d/ data/scene_datasets/hm3d

# # unzip
cd $DATA_Folder
if [ ! -d "objectnav_hm3d_v1" ]; then
    unzip objectnav_hm3d_v1.zip
fi
if [ ! -f "vlobjectnav_hm3d.zip" ]; then
    pip install gdown
    gdown 1fhXwBuGUOhF2jjW0ThtE_6rh_P3YQClj
fi
if [ ! -d "vlobjectnav_hm3d_v4" ]; then
    unzip vlobjectnav_hm3d.zip
fi

# # create datasets folder
cd $VLN_Folder
mkdir -p data/datasets/objectnav/hm3d/
ln -s $DATA_Folder/objectnav_hm3d_v1 data/datasets/objectnav/hm3d/v1
ln -s $DATA_Folder/vlobjectnav_hm3d_v4 data/datasets/vlobjectnav_hm3d

print_ok "Dataset setup complete."