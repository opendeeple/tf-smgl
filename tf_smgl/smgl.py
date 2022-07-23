import os
import pickle
import tensorflow as tf
from tf_smpl.layers import BlendSkinning

class SMGL(tf.keras.layers.Layer):
  def __init__(self, path, **kwargs):
    super(SMGL, self).__init__(**kwargs)
    with open(path, "rb") as __file:
      params = pickle.load(__file)

    with tf.name_scope("smgl"):
      self.poses = tf.constant(
        value=params["pose"],
        shape=[1, 72],
        dtype=self.dtype
      )
      self.shapes = tf.constant(
        value=params["shape"],
        shape=[1, 10],
        dtype=self.dtype
      )
      self.trans = tf.constant(
        value=params["trans"],
        shape=[1, 3],
        dtype=self.dtype
      )
      self.v_template = tf.convert_to_tensor(
        value=params["v_template"],
        name="v_template",
        dtype=self.dtype
      )
      self.faces = tf.convert_to_tensor(
        value=params["f"],
        name="faces",
        dtype=tf.int32
      )
      self.shapedirs = tf.Variable(
        initial_value=params["shapedirs"],
        name="shapedirs",
        dtype=self.dtype
      )
      self.v_indices = tf.convert_to_tensor(
        value=params["indices"],
        name="v_indices",
        dtype=tf.int32
      )
      self.J_transforms = tf.convert_to_tensor(
        value=params["J_transforms"],
        name="J_transforms",
        dtype=self.dtype
      )
      self.J_transforms_inv = tf.linalg.inv(self.J_transforms)
      self.blend_skinning = BlendSkinning(v_weights=params["weights"], dtype=self.dtype)

  def call(self, shape, pose=None, trans=None, body_dict=None, psd=None):
    v_shaped = self.v_template + tf.reshape(
      tensor=tf.matmul(shape, self.shapedirs),
      shape=[-1, self.v_template.shape[0], 3],
      name="v_ssd")

    if pose is None:
      if trans is not None:
        v_shaped += tf.reshape(trans, shape=(-1, 3))[:, tf.newaxis, :]
      return v_shaped

    v_garment = self.skinning(pose, v_shaped=v_shaped, body_dict=body_dict, psd=psd)
    if trans is not None:
      v_garment += tf.reshape(trans, shape=(-1, 3))[:, tf.newaxis, :]
    return v_garment

  def skinning(self, pose, body_dict, v_shaped=None, psd=None):
    if v_shaped is None:
      v_shaped = tf.repeat(self.v_template[tf.newaxis], [pose.shape[0]], axis=0)
    if psd is not None:
      v_shaped = v_shaped + psd
    J_transforms = []
    for i in range(body_dict["J_transforms"].shape[0]):
      J_transforms.append(body_dict["J_transforms"][i] @ self.J_transforms_inv)
    J_transforms = tf.convert_to_tensor(
      value=J_transforms,
      dtype=self.dtype
    )
    v_garment = self.blend_skinning(v_shaped, J_transforms)
    return v_garment

  def fix_collisions(self, vc, vb, nb, v_indices, eps=0.002):
    vb = tf.gather(vb, v_indices, batch_dims=1)
    nb = tf.gather(nb, v_indices, batch_dims=1)

    penetrations = tf.reduce_sum(nb * (vc - vb), axis=2) - eps
    penetrations = tf.math.minimum(penetrations, eps)

    corrective_offset = -tf.multiply(nb, penetrations[:, :, tf.newaxis])
    vc_fixed = vc + corrective_offset
    return vc_fixed

  def save(self, path):
    params = {
      "indices": self.v_indices.numpy(),
      "shapedirs": self.shapedirs.numpy(),
      "weights": self.blend_skinning.v_weights.numpy(),
      "f": self.faces.numpy(),
      "v_template": self.v_template.numpy()
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as __file:
      pickle.dump(params, __file)
