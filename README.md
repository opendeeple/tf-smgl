# TensorFlow implementation of SMGL (A Skinned Multi-Garment Linear Model)

## Installation
```
$ pip install git+https://github.com/opendeeple/tf-smgl.git
```

## Clone for local usage
```
$ git clone https://github.com/opendeeple/tf-smgl.git
$ cd tf-smgl
```

## Build SMGL model
```
$ smgl-build --config <path-to-config:config> --path <path-to-save:pkl>
```
## Local SMGL model builder
```
$ python tf_smgl/bin/build_smgl.py --config <path-to-config:config> --path <path-to-save:pkl>
```

## Run SMGL model
```
$ smgl --path <path-to-smgl:pkl> --smpl <path-to-smpl:pkl> --motion <path-to-poses:npz> --batch-size <batch-size:int=16> --shape-range <shape-range:int=0> --output <path-to-save:pc2>
```
## Run local SMGL model
```
$ python tf_smgl/bin/run_smgl.py --path <path-to-smgl:pkl> --smpl <path-to-smpl:pkl> --motion <path-to-poses:npz> --batch-size <batch-size:int=16> --shape-range <shape-range:int=0> --output <path-to-save:pc2>
```

## Usage
```py
import tensorflow as tf
from tf_smpl import SMPL
from tf_smgl import SMGL

smpl = SMPL("<path-to-smpl:pkl>")
smgl = SMGL("<path-to-smgl:pkl>")
# calculate SMPL vertices
v_body, body_dict = smpl(
  shapes=betas,
  poses=poses,
  trans=trans,
  includes=["J_transforms"]
)
# calculate body normals
v_normals = smpl.normals(v_body)
# calculate SMGL vertices
v_garment = smgl(
  shapes=betas, 
  poses=poses, 
  trans=trans, 
  body_dict=body_dict
)
# Calculate neighbours of SMPL with Outfit
v_indices = smpl.neighbours(v_body, v_garment)
# Fix collision errors
v_fixed = smgl.fix_collisions(v_garment, v_body, v_normals, v_indices=v_indices)
```
