import pandas as pd
import  numpy as np
from keras import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, RMSprop
import optuna
import pickle

df=pd.read_csv("spam_ham_dataset.csv")
df=df.drop("Unnamed: 0",axis=1)
#df.to_csv("spam_ham_dataset.csv", index=False)


print(df.columns)
print(df.isnull().sum())
x=np.array(df["text"])
y=np.array(df["label_num"])

X_train, X_val, y_train, y_val = train_test_split(
    x, y, test_size=0.2, random_state=42
)

vectorizer = CountVectorizer()
x_train_vect = vectorizer.fit_transform(X_train)
x_val_vect = vectorizer.transform(X_val)

def objective(trial):
    # Hyperparameters
    units1 = trial.suggest_int("units1", 32, 256, step=32)
    use_second_layer = trial.suggest_categorical("use_second_layer", [True, False])
    units2 = trial.suggest_int("units2", 32, 128, step=32) if use_second_layer else 0
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "rmsprop"])
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)

    # Build model
    model = Sequential()
    model.add(Dense(units=units1, activation="relu", input_shape=(x_train_vect.shape[1],)))
    if use_second_layer:
        model.add(Dense(units=units2, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    # Optimizer
    optimizer = Adam() if optimizer_name == "adam" else RMSprop()
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

    # Train model
    history = model.fit(
        x_train_vect.toarray(), y_train,
        validation_data=(x_val_vect.toarray(), y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        verbose=0  # silent
    )

    # Return last epoch validation accuracy
    val_acc = history.history["val_accuracy"][-1]
    return val_acc

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)
print("Best hyperparameters:", study.best_trial.params)
best_params = study.best_trial.params
model = Sequential()

model.add(Dense(best_params["units1"], activation="relu", input_shape=(x_train_vect.shape[1],)))
if best_params["use_second_layer"]:
    model.add(Dense(best_params["units2"], activation="relu"))
model.add(Dense(1, activation="sigmoid"))

optimizer = Adam() if best_params["optimizer"] == "adam" else RMSprop()
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

# Train best model
model.fit(
    x_train_vect.toarray(), y_train,
    validation_data=(x_val_vect.toarray(), y_val),
    epochs=best_params["epochs"],
    batch_size=best_params["batch_size"],
    verbose=2
)


model.save("spam_model.keras",save_format="keras")

# save vectorizer
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)


