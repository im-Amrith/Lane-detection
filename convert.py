import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dropout, UpSampling2D
from keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D
from keras.layers import BatchNormalization
import h5py

def create_model(input_shape=(80, 160, 3), pool_size=(2, 2)):
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    
    # Conv Layer 1
    model.add(Conv2D(8, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Conv1'))
    
    # Conv Layer 2
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Conv2'))
    
    # Pooling 1
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Conv Layer 3
    model.add(Conv2D(16, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Conv3'))
    model.add(Dropout(0.2))
    
    # Conv Layer 4
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Conv4'))
    model.add(Dropout(0.2))
    
    # Conv Layer 5
    model.add(Conv2D(32, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Conv5'))
    model.add(Dropout(0.2))
    
    # Pooling 2
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Conv Layer 6
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Conv6'))
    model.add(Dropout(0.2))
    
    # Conv Layer 7
    model.add(Conv2D(64, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Conv7'))
    model.add(Dropout(0.2))
    
    # Pooling 3
    model.add(MaxPooling2D(pool_size=pool_size))
    
    # Upsample 1
    model.add(UpSampling2D(size=pool_size))
    
    # Deconv 1
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Deconv1'))
    model.add(Dropout(0.2))
    
    # Deconv 2
    model.add(Conv2DTranspose(64, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Deconv2'))
    model.add(Dropout(0.2))
    
    # Upsample 2
    model.add(UpSampling2D(size=pool_size))
    
    # Deconv 3
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Deconv3'))
    model.add(Dropout(0.2))
    
    # Deconv 4
    model.add(Conv2DTranspose(32, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Deconv4'))
    model.add(Dropout(0.2))
    
    # Deconv 5
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Deconv5'))
    model.add(Dropout(0.2))
    
    # Upsample 3
    model.add(UpSampling2D(size=pool_size))
    
    # Deconv 6
    model.add(Conv2DTranspose(16, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Deconv6'))
    
    # Final layer
    model.add(Conv2DTranspose(1, (3, 3), padding='valid', strides=(1,1), activation='relu', name='Final'))
    
    return model

print("TensorFlow version:", tf.__version__)

try:
    # Create the model with the same architecture
    print("Creating model...")
    model = create_model()
    
    # Load weights from the old model
    print("Loading weights...")
    model.load_weights('full_CNN_model.h5')
    
    # Save the model in the new format
    print("Saving model...")
    model.save('converted_CNN_model.h5')
    print("Model successfully converted and saved as 'converted_CNN_model.h5'")
except Exception as e:
    print(f"Error: {str(e)}")
    print("Failed to convert the model.")
