import argparse
import os
import numpy as np
import tensorflow as tf
from model import make_generator,make_discriminator
import datetime
import glob
from utils import run_from_ipython, generate_and_save_images, save_gif

ipython = run_from_ipython()

if ipython:
    from IPython import display


parser = argparse.ArgumentParser(description="train")
parser.add_argument("--output_dir", type=str, default=".",help="output_dir")
parser.add_argument("--epoch",type=int,default=100,help="Epochs")
parser.add_argument("--lr_gen",type=float,default=0.00005,help="Learning rate of generator")
parser.add_argument("--lr_dis",type=float,default=0.00005,help="Learning rate of discriminator")
parser.add_argument("--batch_size",type=int,default=64,help="Batch_Size")
parser.add_argument("--n_critics",type=int,default=5,help="number of Critics")
parser.add_argument("--c",type=float,default=0.01,help="Gradint Clipping Value")
parser.add_argument("--l_depth",type=int,default=100,help="latent depth")
parser.add_argument("--num_exp",type=int,default=16,help="number to example to generator")

args = parser.parse_args()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = f"{args.output_dir}/logs/gradient-tape"+current_time+"/train"
generator_dir = train_log_dir+"/generator"
discriminator_dir = train_log_dir+"/discriminator"
disc_summary_writer=tf.summary.create_file_writer(discriminator_dir)
gen_summary_writer=tf.summary.create_file_writer(generator_dir)

BUFFER_SIZE = 60000
seed = tf.random.normal([args.num_exp, args.l_depth])

# Loading  MNIST_Dataset
(train_images, train_labels),(_,_) = tf.keras.datasets.mnist.load_data()

# Preparing and Normalising Dataset
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(args.batch_size)

generator = make_generator()
discriminator = make_discriminator()

losses = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake):
    loss = losses(tf.ones_like(fake),fake)
    return loss

def discriminator_loss(real,fake):
    true_score = losses(tf.ones_like(true_output),true_output)
    fake_score = losses(tf.zeros_like(fake_ouput2),fake_ouput2)
    loss = true_score+fake_score
    return loss

generator_optimizer = tf.keras.optimizers.RMSprop(lr=args.lr_gen)
discriminator_optimizer = tf.keras.optimizers.RMSprop(lr=args.lr_dis)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
ckpt_manager = tf.train.CheckpointManager(checkpoint,args.output_dir,checkpoint_name='model.ckpt',max_to_keep=3)


@tf.function
def train_dis_step(real,fake):
    with tf.GradientTape() as disc_tape:
        generated_image = generator(fake,training=True)
        true_output = discriminator(real,training=True)
        fake_ouput  = discriminator(generted_image,training=True)
        disc_loss = discriminator_loss(true_output,fake_output)
        grad_disc = disc_tape.gradient(disc_loss,discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(grad_disc,discriminator.trainable_variables))
    
    return disc_loss

@tf.function
def train_gen_step(fake):
    with tf.GraidientTape() as gen_tape:
        fake_output = generator(fake,training = True)
        gen_loss = generator_loss(fake_output)
        grad_gen = gen_tape.gradient(gen_loss,generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(grad_gen,generator.trainable_variables))
                
    for w in discriminator.trainable_variable:
          w.assign(tf.clip_by_value(w,-args.c,args.c))
    
    return gen_loss    

def main():
    for i in range(1,args.epoch+1):
        disc_loss = 0
        for n in range(1,args.n_critics+1):
             
            for x in train_dataset.take(1):
                fake = np.random.normal(size=[args.batch_size,args.l_depth]).astype(np.float32)
                disc_loss = train_dis_step(x,fake)
                
                with disc_summary_writer.as_default():
                    tf.summary.scalar("discriminator_loss",disc_loss,step=i)
            
            print(f"Epoch:{i} step=>{n} : disciminator_loss:{disc_loss}")

        gene_loss = train_gen_step(fake)
        with gen_summary_writer.as_default():
            tf.summary.scalar('generator_loss',gene_loss, step =i)
        
        print(f'Epoch {i} results: Discriminator Loss: {disc_loss}' )
        print(f'Generator Loss: {gene_loss}')
       
        if i % 10 == 0:
              ckpt_manager.save()
       
        if ipython:
            display.clear(wait=True)  
        generate_and_save_images(generator, i , outdir = args.outdir)
    
    if ipython:
        display.clear_output(wait=True)
    generate_and_save_images(generator, i, outdir = args.outdir)
                  
   
main()

    
