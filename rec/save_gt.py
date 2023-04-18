from rec.dataloader import get_data_models, fetch_pcl, get_data_util
import os

params = lambda x: 0

# data
params.data_dir = 'D:\\data\\images\\data'
params.num_views = 10
params.batch_size = 32
params.category = 'table'
params.num_points = 1024
params.save_dir = 'D:\\data\\images\\data\\pcl_gt\\table'

class OBJ:
  def __init__(self, vertices, lines_index):

    self.vertices = vertices
    self.lines = [[line[0], line[1]] for line in lines_index]

  def save_obj(self, output_dir):
    with open(output_dir, 'w') as f:
      f.write('# OBJ file\n')
      for v in self.vertices:
        f.write('v {0} {1} {2}\n'.format(v[0], v[1], v[2]))
      for l in self.lines:
        f.write('l {0} {1}\n'.format(int(l[0]), int(l[1])))

# train_models, train_pair_indices = get_data_models(params.data_dir, 'aero', params.num_views, 'train')
# val_models, val_pair_indices = get_data_models(params.data_dir, 'aero', params.num_views, 'val')


def save_pcl_model(data_dir, save_dir, category, split='train'):
    models = get_data_util(data_dir, category, split)
    for model in models:
        filename = os.path.basename(model)
        pcl = fetch_pcl(model)
        obj = OBJ(pcl, [])
        obj.save_obj(os.path.join(save_dir, split, filename + '.obj'))

if __name__ == '__main__':
    #save_pcl_model(params.data_dir, params.save_dir, params.category, 'test')
    save_pcl_model(params.data_dir, params.save_dir, params.category, 'train')
    save_pcl_model(params.data_dir, params.save_dir, params.category, 'val')




