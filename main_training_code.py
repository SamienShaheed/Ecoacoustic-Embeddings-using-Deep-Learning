


# In[1]:



from google.colab import drive
drive.mount('/content/drive')



# In[2]:



# Change working directory
import os
os.chdir("/content/drive/MyDrive/Google Colab Data/Samien-EAD-Tiny")



# In[3]:



# Read Metadata file of the dataset
import pandas as pd
train = pd.read_csv('time.csv')



# In[4]:



# Check number of rows in Metadata file (number of files in dataset)
len(train)



# In[5]:



# Construct File Path by concatenating Dataset File Name and Class ID
train['relative_path'] = '/' + train['File_Name'].astype(str)
# Create DataFrame with relevant columns
df_train = train[['relative_path']]
train.head()



# In[6]:



df_train.head()



# In[7]:



# Shuffle the Data Frame
df_train = df_train.sample(frac = 1)
df_train



# In[8]:



import math, random
import torch
import numpy as np
import torchaudio
from torchaudio import transforms
import matplotlib.pyplot as plt
from IPython.display import Audio
import sklearn

# Check for pyTorch
print(torch.__version__)
print(torchaudio.__version__)
print(pd.__version__)
print(plt.__version__)
print(np.__version__)
print(sklearn.__version__)
torch.cuda.is_available()



# In[9]:



# ----------------------------
# Load an audio file. Return the signal as a tensor and the sample rate
# ----------------------------
class AudioUtil():
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file) # sig stores Audio Signal (tensor) and sr stores sample rate (int)

    if(sig.shape[0]>2): # Check if first dimension of tensor is >2
        sig = sig[:1, :] # slice and keep only first channel

    return (sig, sr)

  # ----------------------------
  # Convert the given audio to the desired number of channels
  # ----------------------------
  @staticmethod
  def rechannel(aud, new_channel):
    sig, sr = aud

    if (sig.shape[0] == new_channel):
      # Nothing to do
      return aud

    if (new_channel == 1):
      # Convert from stereo to mono by selecting only the first channel
      resig = sig[:1, :]
    else:
      # Convert from mono to stereo by duplicating the first channel
      resig = torch.cat([sig, sig])

    return ((resig, sr))

  # ----------------------------
  # Since Resample applies to a single channel, we resample one channel at a time
  # ----------------------------
  @staticmethod
  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      # Nothing to do
      return aud

    num_channels = sig.shape[0]
    # Resample first channel
    resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      # Resample the second channel and merge both channels
      retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))

  # ----------------------------
  # Pad (or truncate) the signal to a fixed length 'max_ms' in milliseconds
  # ----------------------------
  @staticmethod
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr//1000 * max_ms

    if (sig_len > max_len):
      # Truncate the signal to the given length
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      # Length of padding to add at the beginning and end of the signal
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      # Pad with 0s
      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig, sr)

  # ----------------------------
  # Shifts the signal to the left or right by some percent. Values at the end
  # are 'wrapped around' to the start of the transformed signal.
  # ----------------------------
  @staticmethod
  def time_shift(aud, shift_limit):
    sig,sr = aud
    _, sig_len = sig.shape
    shift_amt = int(random.random() * shift_limit * sig_len)
    return (sig.roll(shift_amt), sr)

  # ----------------------------
  # Generate a Spectrogram
  # ----------------------------
  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

  # ----------------------------
  # Augment the Spectrogram by masking out some sections of it in both the frequency
  # dimension (ie. horizontal bars) and the time dimension (vertical bars) to prevent
  # overfitting and to help the model generalise better. The masked sections are
  # replaced with the mean value.
  # ----------------------------
  @staticmethod
  def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
    _, n_mels, n_steps = spec.shape
    mask_value = spec.mean()
    aug_spec = spec

    freq_mask_param = max_mask_pct * n_mels
    for _ in range(n_freq_masks):
      aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

    time_mask_param = max_mask_pct * n_steps
    for _ in range(n_time_masks):
      aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

    return aug_spec



# In[10]:



from torch.utils.data import DataLoader, Dataset, random_split, SubsetRandomSampler

data_path = ('/content/drive/MyDrive/Google Colab Data/Samien-EAD-Tiny')

# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
  def __init__(self, df, data_path):
    self.df = df
    self.data_path = str(data_path)
    self.duration = 8000 # in milliseconds (1000ms = 1 second)
    self.sr = 44100
    self.channel = 2
    self.shift_pct = 0.4

  # ----------------------------
  # Number of items in dataset
  # ----------------------------
  def __len__(self):
    return len(self.df)

  # ----------------------------
  # Get i'th item in dataset
  # ----------------------------
  def __getitem__(self, idx):
    # Absolute file path of the audio file - concatenate the audio directory with
    # the relative path
    audio_file = self.data_path + self.df.loc[idx, 'relative_path']

    # Get the Sin, Cos, and Hour Value
    sinValue = self.df.loc[idx, 'Sin_Values']
    cosValue = self.df.loc[idx, 'Cos_Values']

    aud = AudioUtil.open(audio_file)
    # Some sounds have a higher sample rate, or fewer channels compared to the
    # majority. So make all sounds have the same number of channels and same
    # sample rate. Unless the sample rate is the same, the pad_trunc will still
    # result in arrays of different lengths, even though the sound duration is
    # the same.
    reaud = AudioUtil.resample(aud, self.sr)
    rechan = AudioUtil.rechannel(reaud, self.channel)

    dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
    shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
    sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
    aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

    return aug_sgram, torch.tensor([sinValue, cosValue])



# In[11]:



import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import init
from torchsummary import summary

class AudioRegressor(nn.Module):

    def __init__(self):
        super().__init__()
        conv_layers = []

        # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(4)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # Second Convolution Block
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # Third Convolution Block
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # Fourth Convolution Block
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # Fifth Convolution Block
        self.conv5 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=(1, 1))
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv5.weight, a=0.1)
        self.conv5.bias.data.zero_()
        conv_layers += [self.conv5, self.relu5, self.bn5]

        # Sixth Convolution Block
        self.conv6 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=(1, 1))
        self.relu6 = nn.ReLU()
        self.bn6 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv6.weight, a=0.1)
        self.conv6.bias.data.zero_()
        conv_layers += [self.conv6, self.relu6, self.bn6]

        # Seventh Convolution Block
        self.conv7 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=(1, 1))
        self.relu7 = nn.ReLU()
        self.bn7 = nn.BatchNorm2d(128)
        init.kaiming_normal_(self.conv7.weight, a=0.1)
        self.conv7.bias.data.zero_()
        conv_layers += [self.conv7, self.relu7, self.bn7]

        # Eighth Convolution Block
        self.conv8 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=(1, 1))
        self.relu8 = nn.ReLU()
        self.bn8 = nn.BatchNorm2d(256)
        init.kaiming_normal_(self.conv8.weight, a=0.1)
        self.conv8.bias.data.zero_()
        conv_layers += [self.conv8, self.relu8, self.bn8]

        # Wrap the Convolutional Blocks
        self.conv = nn.Sequential(*conv_layers)

        # Linear Classifier
        self.ap = nn.AdaptiveMaxPool2d(output_size=1)
        self.lin = nn.Linear(in_features=256, out_features=2)
        self.output=nn.Tanh()

    def forward(self,input_data):
        x = self.conv(input_data)
        x = self.ap(x)
        x = x.view(x.shape[0],-1)
        x = self.lin(x)
        output = self.output(x)

        return output



# In[12]:



from sklearn.model_selection import KFold

# Create the model and put it on the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(42)

# Loss functions
mse_fn = nn.MSELoss() # Mean Squared Error
mae_fn = nn.L1Loss() # Mean Absolute Error

# Parameters
num_epochs=25
batch_size=32
learning_rate = 0.0001

trainds = SoundDS(train, data_path)



# In[13]:



import numpy as np
import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------
# Training Loop
# --------------------------------
def train_epoch_regression(model, device, dataloader, mse_fn, mae_fn, optimizer):
    train_mse_loss = 0.0
    train_mae_loss = 0.0

    all_predictions = []
    all_targets = []

    model.train()

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()

        predictions = model(images)

        mse_loss = mse_fn(predictions.float(), targets.float())
        mae_loss = mae_fn(predictions.float(), targets.float())

        mse_loss.backward()
        optimizer.step()

        train_mse_loss += mse_loss.item() * images.size(0)
        train_mae_loss += mae_loss.item() * images.size(0)

        # Convert predictions, targets and activations to CPU and numpy, then store
        all_predictions.append(predictions.cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())

    return train_mse_loss, train_mae_loss, all_predictions, all_targets

# --------------------------------
# Validation Loop
# --------------------------------
def valid_epoch_regression(model, device, dataloader, mse_fn, mae_fn):
    valid_mse_loss = 0.0
    valid_mae_loss = 0.0

    all_predictions = []
    all_targets = []

    model.eval()

    for images, targets in dataloader:
        images, targets = images.to(device), targets.to(device)

        predictions = model(images)

        mse_loss = mse_fn(predictions.float(), targets.float())
        mae_loss = mae_fn(predictions.float(), targets.float())

        valid_mse_loss += mse_loss.item() * images.size(0)
        valid_mae_loss += mae_loss.item() * images.size(0)

        # Convert predictions and targets to CPU and numpy, then store
        all_predictions.append(predictions.cpu().detach().numpy())
        all_targets.append(targets.cpu().detach().numpy())

    return valid_mse_loss, valid_mae_loss, all_predictions, all_targets



# In[14]:



# --------------------------------
# Convert Sine & Cosine to Hours
# --------------------------------
def convert_trig_to_hours(sinValues, cosValues):

  timeInHours_list = []

  for i in range(len(sinValues)):
    # Convert sin and cos values back to readable time format
    angle_rad = np.arctan2(sinValues[i], cosValues[i])
    timeInHours = (angle_rad * 24) / (2 * np.pi)

    # Adjust for negative values
    if timeInHours < 0:
        timeInHours += 24

    timeInHours_list.append(timeInHours)

  return np.array(timeInHours_list)

# --------------------------------
# Calculate MAE in hourly units
# --------------------------------
def mae_in_hours(targetSinValues, predictedSinValues, targetCosValues, predictedCosValues):
  predicted_time_list = convert_trig_to_hours(predictedSinValues, predictedCosValues)
  actual_time_list = convert_trig_to_hours(targetSinValues, targetCosValues)

  absolute_errors = np.abs(np.array(actual_time_list) - np.array(predicted_time_list))
  hourlyMAE = np.mean(absolute_errors)

  return hourlyMAE



# In[15]:



from torch.utils.data import random_split

# Define the size of the train and validation datasets
train_idx = int(0.8 * len(trainds))  # 80% of the dataset for training
val_idx = len(trainds) - train_idx  # 20% of the dataset for validation

# Split your dataset
train_dataset, val_dataset = random_split(trainds, [train_idx, val_idx])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Capture a list of predicted and actual values
predictedValues = []
actualValues = []

# Start the model
model = AudioRegressor()
model.to(device)

# Change optimizer accordingly
optimizer = torch.optim.Rprop(model.parameters(),lr=learning_rate)

print("Training Samples: {} | Testing Samples: {} | Learning Rate: {}".format(len(train_loader.sampler),
                                                                              len(test_loader.sampler),
                                                                              learning_rate))

history = {'train_mse_loss': [], 'train_mae_loss': [],'test_mse_loss':[],'test_mae_loss':[]}

for epoch in range(num_epochs):
  train_mse_loss, train_mae_loss, trainPredictedValues, trainTargetValues, = train_epoch_regression(model, device, train_loader, mse_fn, mae_fn, optimizer)
  test_mse_loss, test_mae_loss, testPredictedValues, testTargetValues = valid_epoch_regression(model, device, test_loader, mse_fn, mae_fn)

  # Concatenate all predictions and targets
  trainPredictions = np.concatenate(trainPredictedValues, axis=0)
  trainTargets = np.concatenate(trainTargetValues, axis=0)

  # Concatenate all predictions and targets
  testPredictions = np.concatenate(testPredictedValues, axis=0)
  testTargets = np.concatenate(testTargetValues, axis=0)

  train_mse_loss = train_mse_loss / len(train_loader.sampler)
  trainMAE = mae_in_hours(trainTargets[:, 0], trainPredictions[:, 0], trainTargets[:, 1], trainPredictions[:, 1])

  test_mse_loss = test_mse_loss / len(test_loader.sampler)
  testMAE = mae_in_hours(testTargets[:, 0], testPredictions[:, 0], testTargets[:, 1], testPredictions[:, 1])

  print("Epoch:{}/{} AVG Train MAE Loss: {:.3f} hours | AVG Test MAE Loss: {:.3f} hours".format(epoch + 1,
                                                                                                num_epochs,
                                                                                                trainMAE,
                                                                                                testMAE))

history['train_mse_loss'].append(train_mse_loss)
history['train_mae_loss'].append(trainMAE)
history['test_mse_loss'].append(test_mse_loss)
history['test_mae_loss'].append(testMAE)

# Save the model state dictionary
# Rename the file for new saves
torch.save(model.state_dict(), 'model_23-01-2024_v1.pth')