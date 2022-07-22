import os
import sys
import pickle
import argparse
import configparser
import tensorflow as tf
from tf_smpl import SMPL

if __name__ == "__main__" and __package__ is None:
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
  import tf_smgl.bin
  __package__ = "tf_smgl.bin"

from ..utils import load_obj

def args_parser():
  parser = argparse.ArgumentParser(description="Simple training script for training a SNUG network")
  parser.add_argument('--config', help="Config file path", required=True)
  parser.add_argument('--path', help="Output model path", required=True)
  args = parser.parse_args(sys.argv[1:])
  config = configparser.ConfigParser()
  config.read(args.config)
  args.config = config
  return args

def main():
  args = args_parser()
  vertices, faces = load_obj(args.config.get("cloth", "path"))

  with open(args.config.get("smpl", "path"), "rb") as __file:
    params = pickle.load(__file, encoding="latin1")

  garment_shape = tf.constant(
    value=list(map(float, args.config.get("cloth", "shape").split())),
    shape=[1, 10],
    dtype=tf.float32
  )
  garment_pose = tf.constant(
    value=list(map(float, args.config.get("cloth", "pose").split())),
    shape=[1, 72],
    dtype=tf.float32
  )
  garment_trans = tf.zeros(shape=(1, 3), dtype=tf.float32)

  smpl = SMPL(args.config.get("smpl", "path"))
  v_cloth_template, v_cloth_params = smpl(shapes=garment_shape, poses=garment_pose, trans=garment_trans, includes=["J_transforms"])
  v_indices = smpl.neighbours(v_cloth_template, vertices[None])[0]

  params = {
    "shape": garment_shape[0].numpy(),
    "pose": garment_pose[0].numpy(),
    "trans": garment_trans[0].numpy(),
    "indices": v_indices.numpy(),
    "J_transforms": v_cloth_params["J_transforms"][0].numpy(),
    "shapedirs": tf.gather(params["shapedirs"], v_indices).numpy(),
    "weights": tf.gather(params["weights"], v_indices).numpy(),
    "f": faces.numpy(),
    "v_template": vertices.numpy()
  }
  params["shapedirs"] = params["shapedirs"].reshape([-1, params["shapedirs"].shape[-1]]).T

  os.makedirs(os.path.dirname(args.path), exist_ok=True)
  with open(args.path, "wb") as __file:
    pickle.dump(params, __file)

if __name__ == "__main__":
  main()
