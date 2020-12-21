from cnn_model import CNN_Model, train_test_split,plt
from utils import normalize_data, denormalize_data,np, save_model_to_json
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

num_epochs = 100
json_file_name = 'pmus_cnn'

flow = np.load('flow200000.npy')
volume = np.load('volume200000.npy')
paw = np.load('paw200000.npy')
resistances = np.load('rins200000.npy')
capacitances = np.load('capacitances200000.npy')

flow = flow.T
volume = volume.T
paw = paw.T
resistances = resistances.T
capacitances = capacitances.T

num_examples = flow.shape[0]
num_samples = flow.shape[1]

(min_flow, max_flow, flow_norm) = normalize_data(flow)
(min_volume, max_volume, volume_norm) = normalize_data(volume)
(min_paw, max_paw, paw_norm) = normalize_data(paw)
(min_resistance, max_resistance, resistances_norm) = normalize_data(resistances)
(min_capacitance, max_capacitance, capacitance_norm) = normalize_data(capacitances)

input_data = np.zeros((num_examples, num_samples, 3))
input_data[:, :, 0] = flow_norm
input_data[:, :, 1] = volume_norm
input_data[:, :, 2] = paw_norm
output_data = np.concatenate((resistances_norm, capacitance_norm), axis=1)
indices = np.arange(num_examples)

input_train, input_test, output_train, output_test, indices_train, indices_test = \
    train_test_split(input_data, output_data, indices, test_size=0.2, shuffle=True, random_state=0)


model = CNN_Model(num_samples,input_volume = 3).get_model()

es_callback = EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True)

history = model.fit(input_train, output_train, epochs=num_epochs, verbose=1,
                    validation_data=(input_test, output_test) , callbacks=[es_callback])


save_model_to_json(model,json_file_name)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


