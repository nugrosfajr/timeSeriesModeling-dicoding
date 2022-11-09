"""
Proyek Kedua : Membuat Model Machine Learning dengan Data Time Series
Original file is located at
    https://colab.research.google.com/drive/1HECnWIL-O9SfZpr_y-lhP3lswQxkL8Il
Author: Fajar Ari Nugroho_1494037162101-489
"""

! pip install -q kaggle
from google.colab import files
files.upload()
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
! kaggle datasets download -d shaneysze/new-york-city-daily-temperature-18692021 -f nyc_temp_1869_2021.csv
! unzip nyc_temp_1869_2021.csv.zip
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.layers import LSTM, Dense, Dropout


df = pd.read_csv('/content/nyc_temp_1869_2021.csv', encoding= 'unicode_escape')
df
df.info()


df['TMAX'] /= 10
df['TMIN'] /= 10
df['MEAN'] = df[['TMAX', 'TMIN']].astype(float).mean(axis=1)
df['MM/DD/YYYY'] = pd.to_datetime(df['MM/DD/YYYY'])

df = df.rename(columns={
    "MM/DD/YYYY":"date",
    "MEAN":"temp_avg"})
df.drop(['Unnamed: 0', 'YEAR', 'MONTH', 'DAY', 'TMAX', 'TMIN'], axis=1, inplace=True)
df.tail(5)


dates = df['date'].values
temp = df['temp_avg'].values

plt.figure(figsize=(15,7))
plt.plot(dates, temp)

plt.title('New York City Average Temperature',
          fontsize=20)
plt.ylabel('Temperature')
plt.xlabel('Datetime')
df.dtypes


suhu = df['temp_avg'].values
suhu

suhu_new = suhu.reshape(-1, 1)

scaler = StandardScaler()                            
scaler.fit(suhu_new)
suhu = scaler.transform(suhu_new)

temp = suhu.flatten()
temp


X_train, X_valid, y_train, y_valid = train_test_split(temp, dates, train_size=0.8, test_size=0.2, shuffle=False)
print('Total Train: ',len(X_train))
print('Total Validation: ',len(X_valid))


def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  series = tf.expand_dims(series, axis=-1)
  ds = tf.data.Dataset.from_tensor_slices(series)
  ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
  ds = ds.flat_map(lambda w: w.batch(window_size + 1))
  ds = ds.shuffle(shuffle_buffer)
  ds = ds.map(lambda w: (w[:-1], w[-1:]))
  return ds.batch(batch_size).prefetch(1)


train_set = windowed_dataset(X_train, window_size=60, batch_size=100, shuffle_buffer=1000)
valid_set = windowed_dataset(X_valid, window_size=60, batch_size=100, shuffle_buffer=1000)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.LSTM(60),
    tf.keras.layers.Dense(30, activation='relu'),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1),
])


threshold_mae = (temp.max() - temp.min()) * 10/100
print(threshold_mae)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('mae') < threshold_mae and logs.get('val_mae') < threshold_mae):
      print('\nEpoch', epoch, '\nGreat!, MAE of your model has reach <10% of data scale', 'training is already stop!')
      self.model.stop_training = True

optimizer = tf.keras.optimizers.SGD(lr=1.0000e-04, momentum=0.9)

model.compile(
    loss=tf.keras.losses.Huber(),
    optimizer=optimizer,
    metrics=["mae"]
)


tf.keras.backend.set_floatx('float64')

history = model.fit(
    train_set,
    validation_data=valid_set,
    epochs=100,
    callbacks = [myCallback()]
)



plt.figure(figsize=(14, 5))
# Accuracy Plot
plt.subplot(1, 2, 1)
mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(len(mae))
plt.plot(epochs, mae, label='Training mae')
plt.plot(epochs, val_mae, label='Validation mae')
plt.legend(loc='upper right')
plt.title('Training and Validation Accuracy')

# Loss Plot
plt.subplot(1, 2, 2)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(mae))
plt.plot(epochs, loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()