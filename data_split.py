import glob
import numpy as np
has_glass_files = glob.glob('/home/eric/CelebA/Img/img_celeba.7z/hasGlassAlign/*.jpg')
no_glass_files = glob.glob('/home/eric/CelebA/Img/img_celeba.7z/noGlassAlign/*.jpg')

print(len(has_glass_files),len(no_glass_files))

has_glass_files = glob.glob('/home/eric/CelebA/Img/img_celeba.7z/hasGlassAlign/*.jpg')
no_glass_files = glob.glob('/home/eric/CelebA/Img/img_celeba.7z/noGlassAlign/*.jpg')
np.random.shuffle (has_glass_files)
np.random.shuffle(no_glass_files)
m = len(has_glass_files)
n = len(no_glass_files)
print(m, n, 0.05 * m)
has_glass_files_train = has_glass_files[:int(m)]
has_glass_files_val = has_glass_files[int(0.95 * m):]
no_glass_files = np.random.choice(no_glass_files, m)
no_glass_files_train = no_glass_files[:int(0.95 * m)]
has_glass_files_val = no_glass_files[int(0.95 * m):]