# SegNav:

![Alt text](relative%20/segmented.jpeg?raw=true "Title")
![Alt text](relative%20/architec.jpeg?raw=true "Title")


# Environment


### Step 1: Create Conda Environment

```
conda create -n segnav python=3.7 -y
conda activate segnav
conda install pytorch=1.6.0 torchvision cudatoolkit=10.1 -c pytorch
# or use 
# conda install pytorch=1.10.0 torchvision cudatoolkit=11.3 -c pytorch
```

### Step 2: Installing MMCV (1.3.16)

```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html
# or use
# pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
```
Note: Make sure you mmcv version is compatible with your pytorch and cuda version. In addition, you can specify the MMCV verion (1.3.16).

### Step 3: Installing SegNav
```
git clone https://github.com/VIML-NITA/SegNav.git
cd SegNav
pip install einops prettytable
pip install -e . 
```


# Get Started

In this section, we explain the data generation process and how to train and test our network.

## Data Processing

To be able to run our network, please follow those steps for generating processed data.

### Dataset Download: 

Please go to [RUGD](http://rugd.vision/) and [RELLIS-3D](https://github.com/unmannedlab/RELLIS-3D/blob/main/README.md#annotated-data) (we use the ID annotation instead of color annotation for RELLIS-3D) officail website to download their data. 
<!-- Please structure the downloaded data as follows: -->

Note: Since RELLIS-3D dataset has been updated, please run `python ./tools/process_rellis.py` to remove "pylon_camera_node/", "pylon_camera_node_label_id/" folder, after structure the data as follows:

```
SegNav
├── data
│   ├── rellis
│   │   │── test.txt
│   │   │── train.txt
│   │   │── val.txt
│   │   │── annotation
│   │   │   ├── 00000 & 00001 & 00002 & 00003 & 00004 
│   │   │── image
│   │   │   ├── 00000 & 00001 & 00002 & 00003 & 00004 
│   ├── rugd
│   │   │── test_ours.txt
│   │   │── test.txt
│   │   │── train_ours.txt
│   │   │── train.txt
│   │   │── val_ours.txt
│   │   │── val.txt
│   │   │── RUGD_annotations
│   │   │   ├── creek & park-1/2/8 & trail-(1 & 3-7 & 9-15) & village
│   │   │── RUGD_frames-with-annotations
│   │   │   ├── creek & park-1/2/8 & trail-(1 & 3-7 & 9-15) & village
├── configs
├── tools
...
```

### Dataset Processing: 

In this step, we need to process the groundtruth labels, as well as generating the grouped labels.

For RELLIS-3D dataset, run:

   ```
   python ./tools/convert_datasets/rellis_relabel[x].py
   ``` 

For RUGD dataset, run:

   ```
   python ./tools/convert_datasets/rugd_relabel[x].py
   ``` 

Replease [x] with 4 or 6, to generated data with 4 annotation groups or 6 annotation groups.

## Training

To train a model on RUGD datasets with our methods on 6 groups:
```
python ./tools/train.py ./configs/ours/segnav_group6_rugd.py
```

Please modify `./configs/ours/*` to play with your model and read `./tools/train.py` for more details about training options.

To train a model on multiple GPUs(RUGD, 6 groups, 2 GPUs):
```
./tools/dist_train.sh ./configs/ours/segnav_group6_rugd.py 2
```

## Testing

An example to evaluate our method with 6 groups on RUGD datasets with mIoU metrics:

```
python ./tools/test.py ./trained_models/rugd_group6/segnav_rugd_6.py \
          ./trained_models/rugd_group6/segnav_rugd_6.pth --eval=mIoU
```
Please read `./tools/test.py` for more details.

<!-- To repreduce the papers results, please refer `./trained_models` folder. Please download the trained model [here](https://drive.google.com/drive/folders/1PYn_kT0zBGOIRSaO_5Jivaq3itrShiPT?usp=sharing). -->



# License

This project is released under the [Apache 2.0 license](LICENSE).



