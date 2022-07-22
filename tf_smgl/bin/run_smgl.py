import os
import sys
import argparse
import configparser
from tqdm import tqdm
import tensorflow as tf
from tf_smpl import SMPL

if __name__ == "__main__" and __package__ is None:
  sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
  import tf_smgl.bin
  __package__ = "tf_smgl.bin"

from .. import SMGL
from ..utils import load_motion
from ..utils import save_obj
from ..utils import save_pc2_frames

def args_parser():
  parser = argparse.ArgumentParser(description="Simple training script for training a SNUG network")
  parser.add_argument('--path', help="Path to SMGL model", required=True)
  parser.add_argument('--motion', help="Data file poses.npz", required=True)
  parser.add_argument('--batch-size', help="Batch size", default=16)
  parser.add_argument('--shape-range', help="Shape range to generate", default=0)
  parser.add_argument('--output', help="Output folder name", required=True)
  args = parser.parse_args(sys.argv[1:])
  return args

def main():
  args = args_parser()
  smpl = SMPL("assets/SMPL/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
  smgl = SMGL(args.path)
  v_cloth_template = smpl(poses=smgl.poses, shapes=smgl.shapes, trans=smgl.trans)
  smpl.v_cloth_template = v_cloth_template[0]
  
  pose, trans, trans_vel = load_motion(args.motion)
  data = tf.data.Dataset.from_tensor_slices((pose, trans))
  data = data.batch(batch_size=args.batch_size)

  os.makedirs(args.output, exist_ok=True)
  save_obj(os.path.join(args.output, "body.obj"), smpl.v_cloth_template.numpy(), smpl.faces.numpy())
  save_obj(os.path.join(args.output, "garment.obj"), smgl.v_template.numpy(), smgl.faces.numpy())

  if os.path.isfile(os.path.join(args.output, "body.pc2")):
    os.remove(os.path.join(args.output, "body.pc2"))
    os.remove(os.path.join(args.output, "garment.pc2"))
    os.remove(os.path.join(args.output, "garment_fixed.pc2"))

  for pose, trans in tqdm(data):
    shape = args.shape_range * tf.random.uniform(
      shape=[args.batch_size, 10], minval=-1, maxval=1, dtype=tf.float32)
    
    v_body, body_dict = smpl(shape, pose, trans, ["J_transforms"])
    v_normals = smpl.normals(v_body)
    v_garment = smgl(shape, pose, trans, body_dict)

    v_indices = smpl.neighbours(v_body, v_garment)
    v_fixed = smgl.fix_collisions(v_garment, v_body, v_normals, v_indices=v_indices)

    save_pc2_frames(os.path.join(args.output, "body.pc2"), v_body.numpy())
    save_pc2_frames(os.path.join(args.output, "garment.pc2"), v_garment.numpy())
    save_pc2_frames(os.path.join(args.output, "garment_fixed.pc2"), v_fixed.numpy())

if __name__ == "__main__":
  main()
