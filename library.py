#using word to fact
import gensim
embeddings = gensim.models.Word2Vec.load(namaFileW2Vec)
# atur parameternya disini
model2 = anago.Sequence(char_emb_size=25, word_emb_size=100, char_lstm_units=25,
                       word_lstm_units=100, dropout=0.5, char_feature=True, crf=True,
                       batch_size=20, optimizer='adam', learning_rate=0.001, lr_decay=0.9,
                       clip_gradients=5.0, max_epoch=30, early_stopping=True, patience=3, train_embeddings=True,
                       max_checkpoints_to_keep=5, log_dir=None, embeddings = embeddings)

model2.train(x_train, y_train, x_valid, y_valid)
###

#testing NER ijad
import anago.reader
from pprint import pprint


# weights = "models/ModelAnagoIndo/model_weights.h5"
# params = "models/ModelAnagoIndo/config.json"
# preprocessor = "models/ModelAnagoIndo/preprocessor.pkl"

model = anago.Sequence.load('../models/ModelAnagoIndo')


text = 'Pentas studi teater mata angin 2019 sabtu 4/05/2019 pukul 18 00 wib Gedung Teater STKW Jl Klampis Anom'
words = text.split()
pprint(model.analyze(words))
