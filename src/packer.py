import zipfile
import os
import re

class Packer:
  @staticmethod
  def get_cur_dir():
    return os.path.dirname(os.path.abspath(__file__))

  @staticmethod
  def pack():
    cur_path = sai_packer.get_cur_dir()

    arc_name = os.path.basename(cur_path)
    package_name = arc_name + '_package'
    package_filename = package_name + '.zip'
    filename = os.path.join(cur_path, package_filename)
    excludes = {'__pycache__', '.ipynb_checkpoints'}

    paths = []
    for par_dir, chl_dirs, files in os.walk(cur_path):
      par_dir_base = os.path.basename(os.path.normpath(par_dir))
      if par_dir_base in excludes:
        continue

      arc_path = arc_name + re.sub(cur_path, '', par_dir)
      for file in files:
        if file not in excludes and not file.endswith('zip'):
          paths.append((os.path.join(par_dir, file), os.path.join(arc_path, file)))

    zip_file = zipfile.ZipFile(filename, 'w')
    for i, o in paths:
      zip_file.write(i, o, compress_type=zipfile.ZIP_DEFLATED)

    zip_file.close()
    return os.path.join(cur_path, package_filename)

  @staticmethod
  def register(hs):
    packages = Packer.pack()
    hs.register_resource(packages)
    
  
