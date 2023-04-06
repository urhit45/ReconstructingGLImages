import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import minilensmaker as mil


# Download and unzip the COSMOS data from https://zenodo.org/record/3242143
# then update the path below to point to the folder containing all the files.
catalog = mil.COSMOSCatalog('../data/COSMOS_23.5_training_sample')

magnitude_range = (16, 20)
redshift_range = (0, 1)

x, y = catalog.catalog['mag_auto'], catalog.catalog['zphot'], 
plt.hist2d(x, y,
           bins=(np.linspace(16, 23.5, 50), np.linspace(0, 5, 50)), 
           norm=matplotlib.colors.LogNorm());
plt.colorbar(label='Galaxies / bin')

mil.draw_box(magnitude_range, redshift_range, edgecolor='r')

selection_mask = (
    (magnitude_range[0] <= x) & (x < magnitude_range[1])
    & (redshift_range[0] <= y) & (y < redshift_range[1]))

print(selection_mask.size)
print(selection_mask)
count = np.count_nonzero(selection_mask)
print(count)

n_images = count * 10

md = dict()

# Index in the catalog of the galaxy image to use
md['source_index'] = catalog.sample_indices(n_images, selection_mask)

# Rotation (in radians) to give the source image
md['source_rotation'] = 2 * np.pi * np.random.rand(n_images)

# Einstein radius in arcseconds of the lens
md['theta_E'] = np.random.uniform(0.8, 1.2, n_images)

# Exponent of the lens's power-law mass distribution
md['gamma'] = 2 + 0.1 * np.random.randn(n_images)

# Ellipticity components of the lens
md['e1'], md['e2'] = np.random.uniform(-0.05, 0.05, (2, n_images))

# Location of the lens in the image, in arcseconds
md['center_x'], md['center_y'] = 0.2 * np.random.randn(2, n_images)

# External shear components
md['gamma1'], md['gamma2'] = 0.16 * np.random.randn(2, n_images)

md['fileName'] = np.arange(n_images)

# You can save this to csv, or whatever other output you prefer
csvfile= pd.DataFrame(md)
csvfile["fileName"] = np.arange(csvfile.shape[0])
file = "../data/ProjectData/parameters.csv"


imageDict = csvfile.to_dict('records')

csvfile.head()
# np.save(file,csvfile)
csvfile.to_csv (file, index = False, header=True)


print(np.unique(md['source_index']).size)
all_indices=np.unique(md['source_index'])

n = 200  # for 2 random indices
index_dev = np.random.choice(all_indices.shape[0], n, replace=False)  
dev_indices = all_indices[index_dev]
new_all_indices = np.delete(all_indices, index_dev)
index_test = np.random.choice(new_all_indices.shape[0], n, replace=False)  
test_indices = new_all_indices[index_test]
train_indices = np.delete(new_all_indices, index_test)
print("train: {} dev: {} test:  {}".format(train_indices.size,dev_indices.size,test_indices.size))
print(train_indices.size)

lensmaker = mil.LensMaker(catalog=catalog, pixel_width=0.08)

plot = True
# A dummy 'lens' that does not lens at all.
no_lens = [('SIS', dict(theta_E=0))]

for i in range(n_images):

    source_kwargs = dict(
        catalog_i=md['source_index'][i], 
        phi=md['source_rotation'][i])
    
    unlensed_img = lensmaker.lensed_image(
        lenses=no_lens,
        **source_kwargs)
    
    lensed_img = lensmaker.lensed_image(
        lenses=[
            ('PEMD', {k: md[k][i] for k in 'theta_E gamma e1 e2 center_x center_y'.split()}),
            ('SHEAR', {k: md[k][i] for k in 'gamma1 gamma2'.split()})],
        **source_kwargs)
    # Save to file
    if(md['source_index'][i] in test_indices):
        path = "../data/ProjectData/test"
    elif(md['source_index'][i] in dev_indices):
        path = "../data/ProjectData/dev"
    else:
        path = "../data/ProjectData/train"
                
    unlensed_img_name = "{}/Unlensed/{}.npy".format(path,md['fileName'][i])
    np.save(unlensed_img_name,unlensed_img)
    # Save to file
    lensed_img_name = "{}/Lensed/{}.npy".format(path,md['fileName'][i])
    np.save(lensed_img_name,lensed_img)
    # Here you would save these images in whatever format you prefer.
    # Be aware these images are now arrays of floats; most image formats are made of integers.
    # Unless you use a float-image format (or just save the numpy array),
    # you will either have to commit to some discretization and normalization here.
    plot = False
    if plot:
        f, axes = plt.subplots(ncols=2, figsize=(12, 4))
        
        plt.sca(axes[0])
        mil.plot_image(
            unlensed_img,
            lensmaker.pixel_width)
        plt.title("Unlensed")
        
        plt.sca(axes[-1])
        mil.plot_image(
            lensed_img,
            lensmaker.pixel_width)
        plt.title("Lensed")
        
trainfile = "../data/ProjectData/trainParameters.csv"
devfile = "../data/ProjectData/devParameters.csv"
testfile = "../data/ProjectData/testParameters.csv"
devpara= csvfile.loc[csvfile['source_index'].isin(dev_indices)]
devpara.to_csv (devfile, index = False, header=True)
testpara= csvfile.loc[csvfile['source_index'].isin(test_indices)]
testpara.to_csv (testfile, index = False, header=True)
trainpara= csvfile.loc[csvfile['source_index'].isin(train_indices)]
trainpara.to_csv (trainfile, index = False, header=True)
np.save(file,csvfile)
csvfile.to_csv (file, index = False, header=True)
print(trainpara)

