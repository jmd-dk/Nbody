# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Begin analysis
masterprint('Analysing {} data ...'.format(this_test))

# Read in the three 3D render images
render3D_path   = this_dir + '/output/render3D_snapshot.png'
render3D_0_path = this_dir + '/output/subdir/snapshot_0.png'
render3D_1_path = this_dir + '/output/subdir/snapshot_1.png'
render3D   = plt.imread(render3D_path)
render3D_0 = plt.imread(render3D_0_path)
render3D_1 = plt.imread(render3D_1_path)

# The two identical 3D renders should be exacty equal
if not np.all(render3D_0 == render3D_1):
    abort('The 3D renders "{}" and "{}" are not identical!'.format(render3D_0, render3D_1))

# The dimensions of the images should be as stated in
# render3D.params_0 and render3D.params_1.
for r, path, params in zip((render3D, render3D_0),
                           (render3D_path, render3D_0_path),
                           (this_dir + '/render3D.params_0', this_dir + '/render3D.params_1')):
    module_dict = load_source('params', params).__dict__
    shape = r.shape[:2]
    if shape[0] != shape[1] or shape[0] != module_dict['render3D_resolution']:
        masterprint('done')
        abort('The 3D render "{}" is not of size {}x{}!'
              .format(path, module_dict['render3D_resolution'], module_dict['render3D_resolution']))

# There should be some completely black pixels in the first 3D render
# and some completely white pixels in the second (and third) 3D render
# due to the text.
if not np.any(render3D[:, :, :3] > [0.99]*3):
    abort('The scalefactor text do not seem to '
          'be white on 3D render "{}".'.format(render3D_path))
if not np.any(render3D_0[:, :, :3] < [0.01]*3):
    abort('The scalefactor text do not seem to '
          'be black on 3D render "{}".'.format(render3D_0_path))

# Done analysing
masterprint('done')
