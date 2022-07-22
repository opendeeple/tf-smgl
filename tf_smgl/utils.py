import os
import numpy as np
import tensorflow as tf
from struct import pack, unpack
from scipy.spatial.transform import Rotation as R

def finite_diff_np(x, h, diff=1):
  if diff == 0:
    return x

  v = np.zeros(x.shape, dtype=x.dtype)
  v[1:] = (x[1:] - x[0:-1]) / h

  return finite_diff_np(v, h, diff-1)

def separate_arms(poses, angle=20, left_arm=17, right_arm=16):
  num_joints = poses.shape[-1] //3

  poses = poses.reshape((-1, num_joints, 3))
  rot = R.from_euler('z', -angle, degrees=True)
  poses[:, left_arm] = (rot * R.from_rotvec(poses[:, left_arm])).as_rotvec()
  rot = R.from_euler('z', angle, degrees=True)
  poses[:, right_arm] = (rot * R.from_rotvec(poses[:, right_arm])).as_rotvec()

  poses[:, 23] *= 0.1
  poses[:, 22] *= 0.1

  return poses.reshape((poses.shape[0], -1))

def load_motion(path):
  motion = np.load(path, mmap_mode='r')

  reduce_factor = int(motion['mocap_framerate'] // 30)
  pose = motion['poses'][::reduce_factor, :72]
  trans = motion['trans'][::reduce_factor, :]

  separate_arms(pose)

  swap_rotation = R.from_euler('zx', [-90, 270], degrees=True)
  root_rot = R.from_rotvec(pose[:, :3])
  pose[:, :3] = (swap_rotation * root_rot).as_rotvec()
  trans = swap_rotation.apply(trans)

  trans = trans - trans[0]

  trans_vel = finite_diff_np(trans, 1 / 30)
  return pose.astype(np.float32), trans.astype(np.float32), trans_vel.astype(np.float32)

def load_obj(path, dtype=tf.float32):
  V = []
  F = []
  with open(path, 'r') as __file:
    T = __file.readlines()
  for line in T:
    if line.startswith('v '):
      V.append([float(n) for n in line.replace('v ','').split(' ')])
    elif line.startswith('f '):
      try:
        F.append([int(n) - 1 for n in line.replace('f ','').split(' ')])
      except:
        try:
          F.append([int(n.split('//')[0]) - 1 for n in line.replace('f ','').split(' ')])
        except:
          F.append([int(n.split('/')[0]) - 1 for n in line.replace('f ','').split(' ')])
    elif line.startswith('l '):
      F.append([int(n) - 1 for n in line.replace('l ','').split(' ')])
  vertices, faces = np.array(V, np.float32), np.int32(F)
  return tf.convert_to_tensor(vertices, dtype=dtype), tf.convert_to_tensor(faces, dtype=tf.int32)

def save_obj(path, vertices, faces):
  with open(path, 'w') as __file:
    __file.write('s 1\n')
    for vertex in vertices:
      line = 'v {}\n'.format(' '.join([str(_) for _ in vertex]))
      __file.write(line)
    for face in faces:
      line = 'f {}\n'.format(' '.join([str(_ + 1) for _ in face]))
      if len(face) == 2:
        line = line.replace('f ', 'l ')
      __file.write(line)

def save_pc2(path, V, float16=False):
  if float16: V = V.astype(np.float16)
  else: V = V.astype(np.float32)
  with open(path, 'wb') as __file:
    header_format='<12siiffi'
    header_str = pack(header_format, b'POINTCACHE2\0', 1, V.shape[1], 0, 1, V.shape[0])
    __file.write(header_str)
    __file.write(V.tobytes())

def save_pc2_frames(path, V, float16=False):
  if os.path.isfile(path):
    if float16: V = V.astype(np.float16)
    else: V = V.astype(np.float32)
    with open(path, 'rb+') as __file:
      __file.seek(16)
      nPoints = unpack('<i', __file.read(4))[0]
      assert len(V.shape) == 3 and V.shape[1] == nPoints, 'Inconsistent dimensions: ' + str(V.shape) + ' and should be (-1,' + str(nPoints) + ',3)'
      __file.seek(28)
      nSamples = unpack('<i', __file.read(4))[0]
      nSamples += V.shape[0]
      __file.seek(28)
      __file.write(pack('i', nSamples))
      __file.seek(0, 2)
      __file.write(V.tobytes())
  else: save_pc2(path, V, float16)
