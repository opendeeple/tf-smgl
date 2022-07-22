import setuptools

setuptools.setup(
  name="tf_smgl",
  version="1.0.0",
  description="TensorFlow implementation of SMGL",
  url="https://github.com/opendeeple/tf-smgl",
  author="Firdavs Beknazarov",
  author_email="opendeeple@gmail.com",
  packages=setuptools.find_packages(),
  install_requires = ["tensorflow", "tf_smpl"],
  dependency_links = ["git+ssh://git@github.com/opendeeple/tf-smpl.git"],
  entry_points = {
    "console_scripts": [
      "smgl-build=tf_smgl.bin.build_smgl:main",
      "smgl=tf_smgl.bin.run_smgl:main"
    ]
  }
)
