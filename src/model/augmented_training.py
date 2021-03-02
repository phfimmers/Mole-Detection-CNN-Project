import pickle

import matplotlib.ticker as mtick
from keras.callbacks import EarlyStopping
from sklearn.metrics import roc_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_auc_score

balanced_histories = []
balanced_roc_data = []

for fold, (train_index, val_index) in enumerate(skf.split(train_bal['filepath'], train_bal['label'])):
    training_data, validation_data = train_bal.iloc[train_index], train_bal.iloc[val_index]
    
    train_gen, val_gen = get_generators(training_data, validation_data)
    
    model = get_model((64, 104, 3), tuning = False, l1_l2reg = 0.001)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name = 'auc')])

    es = EarlyStopping(monitor='val_auc',
                   mode='max',
                   verbose=1,
                   patience=10,
                   restore_best_weights=True)
    
    history = model.fit(train_gen,
                        epochs = 1000,
                        verbose = 2,
                        validation_data = val_gen,
                        callbacks = [es])
    
    balanced_histories.append(history.history)

    # save for later application to test set
    model.save_weights(model_folder / f"balanced_k{fold}.h5")
    
    # save ROC curve data for the split
    fpr, tpr, thresholds = roc_curve(validation_data['label'], model.predict(val_gen), pos_label = '1_doctor')
    auc = round(roc_auc_score(validation_data.label == '1_doctor', model.predict(val_gen)), 2)
    
    balanced_roc_data.append((fpr, tpr, auc))

    tf.keras.backend.clear_session()

with open(model_folder / 'balanced_roc_data.pkl', 'wb') as f:
    pickle.dump(balanced_roc_data, f)

# find split/fold with median performance
max_aucs = [max(history['val_auc']) for history in balanced_histories]
median_split = max_aucs.index(sorted(max_aucs)[2])
print(f'median performance on split: {median_split}')

with open(model_folder / 'balanced_histories.pkl', 'wb') as f:
    pickle.dump(balanced_histories, f)
