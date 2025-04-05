# Smpl2bvh for GENEA Leaderboard and BEAT2 dataset
The instructions below guarantee sucessful conversion of [**BEAT2 dataset**](https://paperswithcode.com/dataset/beat2) `.npz` files to `.bvh` files for visualization inside [**BVHView**](https://github.com/TeoNikolov/BVHView/), both supporting the efforts of the [**GENEA Leaderboard**](https://genea-workshop.github.io/leaderboard/).

## Setup
Follow these instructions to setup smpl2bvh:

1. `git clone https://github.com/GENEALeaderboard/smpl2bvh.git`
2. `cd ./smpl2bvh`
3. `git checkout genea_leaderboard`
4. `conda create -n smpl2bvh python=3.6`
5. `conda activate smpl2bvh`
6. `pip install smplx[all] chumpy`
7. Go to [https://smpl-x.is.tue.mpg.de/login.php](https://smpl-x.is.tue.mpg.de/login.php) >> login / register.
8. Go to [https://smpl-x.is.tue.mpg.de/download.php](https://smpl-x.is.tue.mpg.de/download.php) >> download `Download SMPL-X v1.1 (NPZ+PKL, 830 MB) - Use this for SMPL-X Python codebase`
9. Extract `models_smplx_v1_1.zip/models/smplx/*` >> `smpl2bvh/data/smpl/smplx/`

## Usage

To convert BEAT2 dataset `.npz` files to `.bvh` files, run:

    python smpl2bvh.py --poses "<path_to_npz>" --output "<path_to_save_bvh>"

- `--poses` should include `.npz` file extension
- `--output` should include `.bvh` file extension

---

# --- Original Description Below ---

# smpl2bvh
This repository contains an example script to convert from a SMPL model to a bvh file.

<img src="gif/aistpp.gif" align="center"> <br>
The left side of the figure shows the SMPL grand truth and the right side shows the bvh data.
If you want to convert [AMASS](https://amass.is.tue.mpg.de/) to bvh, please refer to [my another repo](https://github.com/KosukeFukazawa/CharacterAnimationTools#13-load-animation-from-amass).

## Notification
This code is MIT licensed, but SMPL requires a separate license.  
Please see [SMPL official website](https://smpl.is.tue.mpg.de/).

## Requirements
You need to download SMPL models in `./data/smpl/smpl/`.  
You need to download [smplx](https://github.com/vchoutas/smplx) too.  
To install from PyPi simply run:
```
    pip install smplx[all]
```

## How to use?
After downloads all requirements, you can use smpl2bvh like this:
```python
    python smpl2bvh.py --gender MALE --poses ${PATH_TO_Y0UR_INPUT} --fps 60 --output ${PATH_TO_SAVE} --mirror
```

`poses` is an `.npz` file or `.pkl` file.  
`.npz` file must contain `rotations` and `trans` as keys.  
`rotations` value is an np.array consisting of [fnum, 24, 3] and `trans` value is the root transition consisting of [fnum, 3]
(fnum means frame number).  

`.pkl` file must contain `smpl_poses` and `smpl_scaling` and `smpl_trans` as keys.  
`smpl_poses` value is an np.array consisting of [fnum, 72] and `smpl_scaling` value is the scaling parameter. `smpl_trans` value is the root transition consisting of [fnum, 3].  
The format of `pkl` file is the same as [AIST++](https://google.github.io/aistplusplus_dataset/) dataset.  
If you check `--mirror` as an argument, the mirrored motion is also saved.  
After processing, you can find bvh file as `--output`.  
For more information, please refer to `smpl2bvh.py`.  

## Reference
`bvh.py` and `quat.py` are based on [Motion Matching](https://github.com/orangeduck/Motion-Matching).
