import GeodisTK
import time
import psutil
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
from PIL import Image


def geodesic_distance_3d(I, S, spacing, lamb, iter):
    '''
    Get 3D geodesic disntance by raser scanning.
    I: input image array, can have multiple channels, with shape [D, H, W] or [D, H, W, C]
       Type should be np.float32.
    S: binary image where non-zero pixels are used as seeds, with shape [D, H, W]
       Type should be np.uint8.
    spacing: a tuple of float numbers for pixel spacing along D, H and W dimensions respectively.
    lamb: weighting betwween 0.0 and 1.0
          if lamb==0.0, return spatial euclidean distance without considering gradient
          if lamb==1.0, the distance is based on gradient only without using spatial distance
    iter: number of iteration for raster scanning.
    '''
    return GeodisTK.geodesic3d_raster_scan(I, S, spacing, lamb, iter)


def demo_geodesic_distance3d():
    # image_path = "../data/"
    # default_prefix = "img3d"
    # extension = ".nii.gz"
    # input_name = image_path + default_prefix + extension
    image_path = "/Users/zyc/Desktop/DESKTOP/MONAILabel_datasets1/myTask03_Breast/images_crop/"
    default_prefix = "Breast_Training_005_ph3"
    extension = ".nii"
    input_name = image_path + default_prefix + extension
    img = sitk.ReadImage(input_name)
    I = sitk.GetArrayFromImage(img)

    spacing_raw = img.GetSpacing()
    spacing = spacing_raw
    # spacing = [spacing_raw[2], spacing_raw[1], spacing_raw[0]]
    I = np.asarray(I, np.float32)   # shape(48, 128, 128)
    # I = I[18:38, 63:183, 93:233]  # shape(20, 120, 140)

    S = np.zeros_like(I, np.uint8)
    slice = 24
    # S[10][60][70] = 1
    S[24][64][64] = 1
    S[24][84][84] = 1

    # t0 = time.time()
    # D1 = GeodisTK.geodesic3d_fast_marching(I, S, spacing)
    t1 = time.time()
    D2 = geodesic_distance_3d(I, S, spacing, 1.0, 4)
    # dt1 = t1 - t0
    dt2 = time.time() - t1
    D3 = geodesic_distance_3d(I, S, spacing, 0.0, 4)
    D4 = geodesic_distance_3d(I, S, spacing, 0.05, 4)
    # print("runtime(s) fast marching {0:}".format(dt1))
    print("runtime(s) raster scan   {0:}".format(dt2))

    # img_d1 = sitk.GetImageFromArray(D1)
    # img_d1.SetSpacing(spacing_raw)
    # sitk.WriteImage(img_d1, image_path + default_prefix + "_dis1" + extension)


    img_d2 = sitk.GetImageFromArray(D2)
    img_d2.SetSpacing(spacing_raw)
    img_d2.SetOrigin(img.GetOrigin())
    img_d2.SetDirection(img.GetDirection())
    sitk.WriteImage(img_d2, image_path + default_prefix + "_dis2" + extension)

    img_d3 = sitk.GetImageFromArray(D3)
    img_d3.SetSpacing(spacing_raw)
    img_d3.SetOrigin(img.GetOrigin())
    img_d3.SetDirection(img.GetDirection())
    sitk.WriteImage(img_d3, image_path + default_prefix + "_dis3" + extension)

    img_d4 = sitk.GetImageFromArray(D4)
    img_d4.SetSpacing(spacing_raw)
    img_d4.SetOrigin(img.GetOrigin())
    img_d4.SetDirection(img.GetDirection())
    sitk.WriteImage(img_d4, image_path + default_prefix + "_dis4" + extension)

    I_sub = sitk.GetImageFromArray(I)
    I_sub.SetSpacing(spacing_raw)
    I_sub.SetOrigin(img.GetOrigin())
    I_sub.SetDirection(img.GetDirection())
    sitk.WriteImage(I_sub, image_path + default_prefix + "_sub" + extension)

    I = I * 255 / I.max()
    I = np.asarray(I, np.uint8)

    I_slice = I[slice]
    # D1_slice = D1[10]
    D2_slice = D2[slice]
    D3_slice = D3[slice]
    D4_slicer = D4[slice]

    plt.figure(figsize=(15,5))

    plt.subplot(1, 4, 1);
    plt.imshow(I_slice, cmap='gray')
    plt.autoscale(False);
    plt.plot([64], [64], 'ro')
    plt.axis('off');
    plt.title('input image')

    # plt.subplot(1, 5, 2);
    # plt.imshow(D1_slice)
    # plt.axis('off');
    # plt.title('fast marching')

    plt.subplot(1, 4, 2);
    plt.imshow(D2_slice)
    plt.axis('off');
    plt.title('ranster scan')

    plt.subplot(1, 4, 3);
    plt.imshow(D3_slice)
    plt.axis('off');
    plt.title('Euclidean distance')

    plt.subplot(1, 4, 4);
    plt.imshow(D4_slicer)
    plt.axis('off');
    plt.title('Mexture of Geodesic \n and Euclidean distance')
    plt.show()


if __name__ == '__main__':
    demo_geodesic_distance3d()