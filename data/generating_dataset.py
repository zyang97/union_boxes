import os
import shutil
import cv2

category = '03636649'
pcl_name = 'pointcloud_1024.npy'
PNG_SOURCE_FILES = ['00.png', '02.png', '04.png', '06.png', '08.png', '10.png', '12.png', '14.png', '16.png', '18.png']
PNG_FILES = ['render_0.png', 'render_1.png', 'render_2.png', 'render_3.png', 'render_4.png', 'render_5.png', 'render_6.png', 'render_7.png', 'render_8.png', 'render_9.png']


image_dir = os.path.join('C:\\Users\\zyang\\Downloads\\ShapeNetRendering\\ShapeNetRendering', category)
pcl_dir = os.path.join('C:\\Users\\zyang\\Downloads\\ShapeNet_pointclouds', category)
save_dir = os.path.join('D:\\data\\images\\data', category)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

filename_list = os.listdir(image_dir)
k = 0
for filename in filename_list:
    print('process object {}'.format(k))
    save_path = os.path.join(save_dir, filename)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # copy pcls
    pcl_path = os.path.join(pcl_dir, filename, pcl_name)
    shutil.copy(pcl_path, os.path.join(save_path, pcl_name))
    # copy images
    imgs_path = os.path.join(image_dir, filename, 'rendering')
    for i, img_name in enumerate(PNG_SOURCE_FILES):
        img_path = os.path.join(imgs_path, img_name)
        ip_image = cv2.imread(img_path)
        ip_image = cv2.resize(ip_image, (64,64))
        for w in range(64):
            for h in range(64):
                if ip_image[w][h][0] == 0 and ip_image[w][h][1] == 0 and ip_image[w][h][2] == 0:
                    ip_image[w][h][0] = 255
                    ip_image[w][h][1] = 255
                    ip_image[w][h][2] = 255
        cv2.imwrite(os.path.join(save_path, PNG_FILES[i]), ip_image)
    k += 1






