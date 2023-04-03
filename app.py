import os
from flask import Flask, render_template, request, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import cv2

# Define allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)

# Define upload folder and allowed file types
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['ALLOWED_EXTENSIONS'] = ALLOWED_EXTENSIONS

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_files():
    # Get the uploaded files
    file1 = request.files['file1']
    file2 = request.files['file2']

    # Check if the files are allowed
    if file1 and allowed_file(file1.filename) and file2 and allowed_file(file2.filename):
        # Save the files to the upload folder
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        file1.save(os.path.join(app.config['UPLOAD_FOLDER'], filename1))
        file2.save(os.path.join(app.config['UPLOAD_FOLDER'], filename2))

        # Run the Python script to generate the new image
        content_path=os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        style_path =os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        # img1 = img1.convert('RGB')
        # img2 = img2.convert('RGB')
        """Import and configure modules"""

        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['figure.figsize'] = (10,10)
        mpl.rcParams['axes.grid'] = False
        
        import numpy as np
        from PIL import Image
        import time
        import functools
        
        # %tensorflow_version 1.x
        import tensorflow as tf
        
        from tensorflow.keras.utils import image_dataset_from_directory as kp_image
        from tensorflow.python.keras import models 
        # from tensorflow.python.keras import losses
        # from tensorflow.python.keras import layers
        # from tensorflow.python.keras import backend as K
        from tensorflow.keras.preprocessing.image import img_to_array
        
        # # Set up some global values here
        # content_path = 'images.jpg'
        # style_path = 'color.jpg'
        
        """# Visualize the input"""
        
        def load_img(path_to_img):
          max_dim = 512
          img = Image.open(path_to_img)
          long = max(img.size)
          scale = max_dim/long
          img = img.resize((round(img.size[0]*scale), round(img.size[1]*scale)), Image.ANTIALIAS)
          
          img = img_to_array(img)
          
          # We need to broadcast the image array such that it has a batch dimension 
          img = np.expand_dims(img, axis=0)
          return img
        
        content = load_img(content_path).astype('uint8')
        style = load_img(style_path).astype('uint8')
        """## Prepare the data"""
        
        def load_and_process_img(path_to_img):
          img = load_img(path_to_img)
          img = tf.keras.applications.vgg19.preprocess_input(img)
          return img
        
        def deprocess_img(processed_img):
          x = processed_img.copy()
          if len(x.shape) == 4:
            x = np.squeeze(x, 0)
          assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                                     "dimension [1, height, width, channel] or [height, width, channel]")
          if len(x.shape) != 3:
            raise ValueError("Invalid input to deprocessing image")
          
          # perform the inverse of the preprocessing step
          x[:, :, 0] += 103.939
          x[:, :, 1] += 116.779
          x[:, :, 2] += 123.68
          x = x[:, :, ::-1]
        
          x = np.clip(x, 0, 255).astype('uint8')
          return x
        
        # Content layer where will pull our feature maps
        content_layers = ['block5_conv2'] 
        
        # Style layer we are interested in
        style_layers = ['block1_conv1',
                        'block2_conv1',
                        'block3_conv1', 
                        'block4_conv1', 
                        'block5_conv1'
                       ]
        
        num_content_layers = len(content_layers)
        num_style_layers = len(style_layers)
        
        def get_model():
        
          # Load our model. We load pretrained VGG, trained on imagenet data
          vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
          vgg.trainable = False
          # Get output layers corresponding to style and content layers 
          style_outputs = [vgg.get_layer(name).output for name in style_layers]
          content_outputs = [vgg.get_layer(name).output for name in content_layers]
          model_outputs = style_outputs + content_outputs
          # Build model 
          return models.Model(vgg.input, model_outputs)
        
        def get_content_loss(base_content, target):
          return tf.reduce_mean(tf.square(base_content - target))
        
        """# Computing style loss"""
        
        
        
        def gram_matrix(input_tensor):
            """Computes the Gram matrix of a given tensor."""
            # Reshape the tensor to have channels first
            channels = int(input_tensor.shape[-1])
            a = tf.reshape(input_tensor, [-1, channels])
            n = tf.shape(a)[0]
            gram = tf.matmul(a, a, transpose_a=True)
            return gram / tf.cast(n, tf.float32)
        
        def get_style_loss(base_style, gram_target):
            """Computes the style loss between a base style tensor and a target Gram matrix."""
            # Get the shape of the base style tensor
            height, width, channels = base_style.get_shape().as_list()
            
            # Compute the Gram matrix of the base style tensor
            gram_style = gram_matrix(base_style)
          
            # Compute the mean squared difference between the Gram matrices
            loss = tf.reduce_mean(tf.square(gram_style - gram_target))
            
            return loss
        
        def get_feature_representations(model, content_path, style_path):
        
          # Load our images in 
          content_image = load_and_process_img(content_path)
          style_image = load_and_process_img(style_path)
          
          # batch compute content and style features
          style_outputs = model(style_image)
          content_outputs = model(content_image)
          
          
          # Get the style and content feature representations from our model  
          style_features = [style_layer[0] for style_layer in style_outputs[:num_style_layers]]
          content_features = [content_layer[0] for content_layer in content_outputs[num_style_layers:]]
          return style_features, content_features
        
        def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
        
          style_weight, content_weight = loss_weights
        
          model_outputs = model(init_image)
          
          style_output_features = model_outputs[:num_style_layers]
          content_output_features = model_outputs[num_style_layers:]
          
          style_score = 0
          content_score = 0
        
          # Accumulate style losses from all layers
          # Here, we equally weight each contribution of each loss layer
          weight_per_style_layer = 1.0 / float(num_style_layers)
          for target_style, comb_style in zip(gram_style_features, style_output_features):
            style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
            
          # Accumulate content losses from all layers 
          weight_per_content_layer = 1.0 / float(num_content_layers)
          for target_content, comb_content in zip(content_features, content_output_features):
            content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
          
          style_score *= style_weight
          content_score *= content_weight
        
          # Get total loss
          loss = style_score + content_score 
          return loss, style_score, content_score
        
        def compute_grads(cfg):
          with tf.GradientTape() as tape: 
            all_loss = compute_loss(**cfg)
          # Compute gradients wrt input image
          total_loss = all_loss[0]
          return tape.gradient(total_loss, cfg['init_image']), all_loss
        
        import IPython.display
        
        def run_style_transfer(content_path, 
                               style_path,
                               num_iterations=1000,
                               content_weight=1e3, 
                               style_weight=1e-2): 
          # We don't need to (or want to) train any layers of our model, so we set their
          # trainable to false. 
          model = get_model() 
          for layer in model.layers:
            layer.trainable = False
          
          # Get the style and content feature representations (from our specified intermediate layers) 
          style_features, content_features = get_feature_representations(model, content_path, style_path)
          gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
          
          # Set initial image
          init_image = load_and_process_img(content_path)
          init_image = tf.Variable(init_image, dtype=tf.float32)
          # Create our optimizer
          opt = tf.optimizers.Adam(learning_rate=5, epsilon=1e-1)
        
          # For displaying intermediate images 
          
          # Store our best result
          best_loss, best_img = float('inf'), None
          
          # Create a nice config 
          loss_weights = (style_weight, content_weight)
          cfg = {
              'model': model,
              'loss_weights': loss_weights,
              'init_image': init_image,
              'gram_style_features': gram_style_features,
              'content_features': content_features
          }
            
          # For displaying
          num_rows = 2
          num_cols = 5
          # display_interval = num_iterations/(num_rows*num_cols)
        
          
          norm_means = np.array([103.939, 116.779, 123.68])
          min_vals = -norm_means
          max_vals = 255 - norm_means   
        
          for i in range(num_iterations):
            grads, all_loss = compute_grads(cfg)
            loss, style_score, content_score = all_loss
            opt.apply_gradients([(grads, init_image)])
            clipped = tf.clip_by_value(init_image, min_vals, max_vals)
            init_image.assign(clipped)
            # end_time = time.time() 
            
            if loss < best_loss:
            # Update best loss and best image from total loss. 
              best_loss = loss
              best_img = deprocess_img(init_image.numpy())     
          return best_img, best_loss
        
        best, best_loss = run_style_transfer(content_path, 
                                             style_path)
        
        best=Image.fromarray(best.astype('uint8')).convert('RGB')
        best = best.resize((500,500))
        # best = best.save("Output.jpg")
        new_filename = 'new_' + filename1 + '_' + filename2
        best.save(os.path.join(app.config['UPLOAD_FOLDER'], new_filename))

        # Display the generated image to the user
        return render_template('display.html', file1=filename1,file2=filename2,filename=new_filename)

    else:
        # Invalid file type
        return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == '__main__':
    app.run(debug=True)
