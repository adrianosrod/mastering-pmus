from cnn_model import CNN_Model, train_test_split,plt
from utils import normalize_data, denormalize_data,np, save_model_to_json
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

num_epochs = 150
plt.rcParams.update({'font.family': 'serif'})

size = 60000
json_file_name = 'pmus__cnn__150_epochs__'+str(size)
flow = np.load('./data/flow'+str(size)+'.npy')
print("flow carregado")
volume = np.load('./data/volume'+str(size)+'.npy')
print("volume carregado")
paw = np.load('./data/paw'+str(size)+'.npy')
print("paw carregado")
resistances = np.load('./data/rins'+str(size)+'.npy')
print("R carregado")
capacitances = np.load('./data/capacitances'+str(size)+'.npy')
print("C carregado")

flow = flow.T
volume = volume.T
paw = paw.T
resistances = resistances.T
capacitances = capacitances.T

print("transposed")

num_examples = flow.shape[0]
num_samples = flow.shape[1]

(min_flow, max_flow, flow) = normalize_data(flow)
(min_volume, max_volume, volume) = normalize_data(volume)
(min_paw, max_paw, paw) = normalize_data(paw)
(min_resistance, max_resistance, resistances) = normalize_data(resistances)
(min_capacitance, max_capacitance, capacitances) = normalize_data(capacitances)

print("normalized data")

input_data = np.zeros((num_examples, num_samples, 3))
input_data[:, :, 0] = flow
input_data[:, :, 1] = volume
input_data[:, :, 2] = paw
output_data = np.concatenate((resistances, capacitances), axis=1)
indices = np.arange(num_examples)

print("input created")


input_train, input_test, output_train, output_test, indices_train, indices_test = \
    train_test_split(input_data, output_data, indices, test_size=0.3, shuffle=False)

input_validation, input_test, output_validation, output_test, indices_validation, indices_test = \
    train_test_split(input_test, output_test, indices_test, test_size=0.5, shuffle=False)

np.save('./data/input_test.npy', input_test)
np.save('./data/output_test.npy', output_test)

print("before CNN")
model = CNN_Model(num_samples,input_volume = 3).get_model()
print("after CNN")

# es_callback = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

history = model.fit(input_train, output_train, epochs=num_epochs, verbose=1,
                    validation_data=(input_validation, output_validation))#, callbacks=[es_callback])


save_model_to_json(model,json_file_name)

# summarize history for loss
plt.rcParams.update({'font.size': 14})

plt.figure()
plt.grid()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
# plt.title('model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('train_val.eps',format='eps')
plt.show()



