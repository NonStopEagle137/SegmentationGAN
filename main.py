import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import datetime
import tensorflow as tf
from loss import *
#from lap_gen import *
from classifier import *
from preprocessing import *

from tqdm import tqdm
    
def main(train_model):	
    @tf.function
    def train_step_scratch(input_image, target, epoch, checkpoint):
      
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

      generator_gradients = gen_tape.gradient(gen_total_loss,
                                              generator.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)

      generator_optimizer.apply_gradients(zip(generator_gradients,
                                              generator.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))

      with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
        tf.summary.scalar('disc_loss', disc_loss, step=epoch)
      return gen_total_loss, disc_loss, gen_gan_loss, gen_l1_loss
      
    @tf.function
    def train_step(input_image, target, real_class, epoch, checkpoint):
      # print(type(real_class))
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generator_output = model(input_image, training=True)
        class_out = generator_output[0]
        gen_output = generator_output[-1]
        real_class = tf.cast(tf.reshape(tf.expand_dims(real_class, axis = 0),(1,2)), dtype = tf.float32)
        classification_loss = loss_object(class_out, real_class)
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
        gen_total_loss = tf.add(gen_total_loss, classification_loss)

      model_gradients = gen_tape.gradient(gen_total_loss,
                                              model.trainable_variables)
      discriminator_gradients = disc_tape.gradient(disc_loss,
                                                   discriminator.trainable_variables)
      model_optimizer.apply_gradients(zip(model_gradients,
                                              model.trainable_variables))
      discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                  discriminator.trainable_variables))
      return gen_total_loss, disc_loss, gen_gan_loss, gen_l1_loss
        
    def fit(train_ds, epochs, test_ds, rcld):
      
      
      for epoch in range(epochs):
        losses_ = np.zeros(4)
        img_count = 0
        start = time.time()
        display.clear_output(wait=True)
        
        for example_combined in test_ds:
          example_input, example_target = example_combined[:,:example_combined.shape[1]//2], example_combined[:,example_combined.shape[1]//2:]
          # print(example_input.shape, example_target.shape)
          img_count += 1
          if img_count % 2 == 0:
            generate_images(model, example_input, example_target)
        print("Epoch: ", epoch)
        # Train
        for n, combined in tqdm(enumerate(train_ds)):
          if rcld[n] == 0:
            real_class = np.array([0, 1])
          elif rcld[n] == 1:
            real_class = np.array([1, 0])
          
          input_image, target = example_combined[:,:example_combined.shape[1]//2], example_combined[:,example_combined.shape[1]//2:]
          losses_ += train_step(np.expand_dims(input_image,0), np.expand_dims(target,0), real_class, epoch, checkpoint)
        print(f'Total Loss : {losses_[0]} \n Discriminator Loss : {losses_[1]} \n Generator Loss : {losses_[2]} \n MAE : {losses_[3]}')
        print("=========================")
        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 2 == 0:
          checkpoint.save(file_prefix=checkpoint_prefix)
        print ('Time taken for epoch {} is {} sec\n'.format(epoch,
                                                            time.time()-start))
      checkpoint.save(file_prefix=checkpoint_prefix)
      
      
    """Some Variables"""
    dataset = []
    
    """Parameters """
    
    EPOCHS = 25
    EPOCHS_PER_LOT = 1
    LOT_SIZE = 5
    PATH = r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\dataset'
    BUFFER_SIZE = 400
    BATCH_SIZE = 3
    OUTPUT_CHANNELS = 3
    LAMBDA = 500
    
    """Optimizers"""
    
    model_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    """Processing the dataset"""
    for name in glob.glob(PATH + '\*.png'):
        dataset.append(name)

    train_dataset = dataset[:450]
    test_dataset = dataset[450:]
    real_class_dataset = list()
    for entry in dataset:
        if entry.split('\\')[-1][0].islower():
            real_class_dataset.append(0)
        elif entry.split('\\')[-1][0].isupper():
            real_class_dataset.append(1)
    
    assert len(real_class_dataset) == len(dataset)
        

    """Loading the dataset into memory"""
    print("[INFO]Loading training and testing data...")
    

    """Base loss object"""
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    clf = seg_cls(num_classes = 2)
    model = clf.generate_model()
    
    #generator = Generator()
    print("Model")
    print("====================")
    model.summary()
    # discriminator = Discriminator()
    
    disc = discriminator_()
    discriminator = disc.get_model()
    
    print("Discriminator")
    print("====================")
    discriminator.summary()


    checkpoint_dir = r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\Checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=model_optimizer,
                                     generator= model,
                                     discriminator=discriminator)

    log_dir= r"C:\Users\Athrva Pandhare\Desktop\New folder (4)\logs"

    summary_writer = tf.summary.create_file_writer(
      log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if train_model == True:
        test_dataset = grab_img_from_names(test_dataset[45:50])
        try:
            
            checkpoint.restore(tf.train.latest_checkpoint(r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\Checkpoints'))
            print("Loaded Checkpoint Successfully...")
            #tf.train.latest_checkpoint(checkpoint_dir)
        except:
            print("No Checkpoint found...Starting from scratch")
        for j in range(EPOCHS):
            for i in range(0,len(train_dataset)-LOT_SIZE, LOT_SIZE):
                train_dataset_ = grab_img_from_names(train_dataset[i:i+LOT_SIZE])
                
                fit(train_dataset_, EPOCHS_PER_LOT, test_dataset, real_class_dataset)
    else:
        checkpoint.restore(tf.train.latest_checkpoint(r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\Checkpoints'))
        generator.save(r'C:\Users\Athrva Pandhare\Desktop\New folder (3)\saved_model\patchwise_generator_model.h5')
    


if __name__ == "__main__":
  main(train_model = True)