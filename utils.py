import matplotlib.pyplot as plt
import imageio
import os
import glob

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def generate_and_save_images(model, j, test_input, outdir, samples = False):
  
  dataset = "MNIST"
  predictions = model(test_input, training=False)
  fig = plt.figure(figsize=(10,10))
  for i in range(predictions.shape[0]):
      plt.subplot(10, 10, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
  if samples == False:
    plt.savefig('{}/assets/{}_at_epoch_{:04d}.png'.format(outdir, dataset, j))
  else:
    plt.savefig(f"{outdir}/assets/{dataset}_cont_dim_{j}.png")
  plt.show()
 
def save_gif(outdir):
    dataset = "MNIST"
    anim_file = f'{outdir}/assets/{dataset}.gif'
    with imageio.get_writer(anim_file, mode='I') as writer:
      filenames = glob.glob(f'{outdir}/assets/{dataset}*.png')
      filenames = sorted(filenames)
      last = -1
      for i,filename in enumerate(filenames):
        frame = 2*i
        if round(frame) > round(last):
          last = frame
        else:
          continue
        image = imageio.imread(filename)
        writer.append_data(image)
      image = imageio.imread(filename)
      writer.append_data(image)
  