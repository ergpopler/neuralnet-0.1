import tensorflow
from tensorflow import keras


data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

word_index = data.get_word_index()

# Sets values for padding, the start, unknown, and unused data in the word index.

word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# Max data size = maxlen if less then maxlen then add padding to fill, if more then maxlen, it will cut the end off.
# Preprocesses data

train_data = (
    keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250))

test_data = (
    keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250))


# Decode_review will turn the integer value of words into the actual words that are human readable.

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


# Setting up neurons/model

'''model = keras.Sequential()
model.add(keras.layers.Embedding(88000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
x_train = train_data[10000:]

y_val = train_labels[:10000]
y_train = train_labels[10000:]

fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_val, y_val), verbose=1)

# Test accuracy of model

results = model.evaluate(test_data, test_labels)

# Save the model
model.save("model.h5")'''

def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


model = keras.models.load_model("model.h5")

with open("test.txt", encoding="utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace(".", "").strip().split(" ")
        encode = review_encode(nline)
        encode = (
            keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250))
        predict = model.predict(encode)
        print(line)
        print(predict[0])

# Show the results

'''test_review = test_data[2]
predict = model.predict([test_review])
print("Review: ")
print(decode_review(test_review))
print("Prediction: " + str(predict[2]))
print("Actual: " + str(test_labels[2]))
print(results)'''



