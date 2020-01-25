import sys
import getopt
import csv
from datetime import datetime
# mempersiapkan library anago got
from GetOldTweets import got3 as got
import numpy as np
import pandas as pd
import nltk
import re
import os
import codecs
from sklearn import feature_extraction
import mpld3

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import nltk.data
import math

# library preprocessing
#menyiapkan environment untuk preprocessing tweet
import csv
import html
import re
import json
import emoji

# -*- coding: utf-8 -*-

#mempersiapkan library anago
import anago
from anago.reader import load_data_and_labels
import random as rn
import random
import gensim.models.keyedvectors as W2V
import tensorflow as tf
from keras import backend as K

# summarize.py
# Luke Reichold - CSCI 4930
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import reuters
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk.data
import math
import re




DOC_ROOT = 'docs/'
DEBUG = False
SUMMARY_LENGTH = 5  # number of sentences in final summary
stop_words = stopwords.words('english')
ideal_sent_length = 20.0

# load sastrawi's Bahasa Indonesia stopwords as variable called 'stopwords'
factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
# extender = ["ada", "adalah", "adanya", "adapun", "agak", "agaknya", "agar", "akan", "akankah", "akhir", "akhiri", "akhirnya", "aku", "akulah", "amat", "amatlah", "anda", "andalah", "antar", "antara", "antaranya", "apa", "apaan", "apabila", "apakah", "apalagi", "apatah", "artinya", "asal", "asalkan", "atas", "atau", "ataukah", "ataupun", "awal", "awalnya", "bagai", "bagaikan", "bagaimana", "bagaimanakah", "bagaimanapun", "bagi", "bagian", "bahkan", "bahwa", "bahwasanya", "baik", "bakal", "bakalan", "balik", "banyak", "bapak", "baru", "bawah", "beberapa", "begini", "beginian", "beginikah", "beginilah", "begitu", "begitukah", "begitulah", "begitupun", "bekerja", "belakang", "belakangan", "belum", "belumlah", "benar", "benarkah", "benarlah", "berada", "berakhir", "berakhirlah", "berakhirnya", "berapa", "berapakah", "berapalah", "berapapun", "berarti", "berawal", "berbagai", "berdatangan", "beri", "berikan", "berikut", "berikutnya", "berjumlah", "berkali-kali", "berkata", "berkehendak", "berkeinginan", "berkenaan", "berlainan", "berlalu", "berlangsung", "berlebihan", "bermacam", "bermacam-macam", "bermaksud", "bermula", "bersama", "bersama-sama", "bersiap", "bersiap-siap", "bertanya", "bertanya-tanya", "berturut", "berturut-turut", "bertutur", "berujar", "berupa", "besar", "betul", "betulkah", "biasa", "biasanya", "bila", "bilakah", "bisa", "bisakah", "boleh", "bolehkah", "bolehlah", "buat", "bukan", "bukankah", "bukanlah", "bukannya", "bulan", "bung", "cara", "caranya", "cukup", "cukupkah", "cukuplah", "cuma", "dahulu", "dalam", "dan", "dapat", "dari", "daripada", "datang", "dekat", "demi", "demikian", "demikianlah", "dengan", "depan", "di", "dia", "diakhiri", "diakhirinya", "dialah", "diantara", "diantaranya", "diberi", "diberikan", "diberikannya", "dibuat", "dibuatnya", "didapat", "didatangkan", "digunakan", "diibaratkan", "diibaratkannya", "diingat", "diingatkan", "diinginkan", "dijawab", "dijelaskan", "dijelaskannya", "dikarenakan", "dikatakan", "dikatakannya", "dikerjakan", "diketahui", "diketahuinya", "dikira", "dilakukan", "dilalui", "dilihat", "dimaksud", "dimaksudkan", "dimaksudkannya", "dimaksudnya", "diminta", "dimintai", "dimisalkan", "dimulai", "dimulailah", "dimulainya", "dimungkinkan", "dini", "dipastikan", "diperbuat", "diperbuatnya", "dipergunakan", "diperkirakan", "diperlihatkan", "diperlukan", "diperlukannya", "dipersoalkan", "dipertanyakan", "dipunyai", "diri", "dirinya", "disampaikan", "disebut", "disebutkan", "disebutkannya", "disini", "disinilah", "ditambahkan", "ditandaskan", "ditanya", "ditanyai", "ditanyakan", "ditegaskan", "ditujukan", "ditunjuk", "ditunjuki", "ditunjukkan", "ditunjukkannya", "ditunjuknya", "dituturkan", "dituturkannya", "diucapkan", "diucapkannya", "diungkapkan", "dong", "dua", "dulu", "empat", "enggak", "enggaknya", "entah", "entahlah", "guna", "gunakan", "hal", "hampir", "hanya", "hanyalah", "hari", "harus", "haruslah", "harusnya", "hendak", "hendaklah", "hendaknya", "hingga", "ia", "ialah", "ibarat", "ibaratkan", "ibaratnya", "ibu", "ikut", "ingat", "ingat-ingat", "ingin", "inginkah", "inginkan", "ini", "inikah", "inilah", "itu", "itukah", "itulah", "jadi", "jadilah", "jadinya", "jangan", "jangankan", "janganlah", "jauh", "jawab", "jawaban", "jawabnya", "jelas", "jelaskan", "jelaslah", "jelasnya", "jika", "jikalau", "juga", "jumlah", "jumlahnya", "justru", "kala", "kalau", "kalaulah", "kalaupun", "kalian", "kami", "kamilah", "kamu", "kamulah", "kan", "kapan", "kapankah", "kapanpun", "karena", "karenanya", "kasus", "kata", "katakan", "katakanlah", "katanya", "ke", "keadaan", "kebetulan", "kecil", "kedua", "keduanya", "keinginan", "kelamaan", "kelihatan", "kelihatannya", "kelima", "keluar", "kembali", "kemudian", "kemungkinan", "kemungkinannya", "kenapa", "kepada", "kepadanya", "kesampaian", "keseluruhan", "keseluruhannya", "keterlaluan", "ketika", "khususnya", "kini", "kinilah", "kira", "kira-kira", "kiranya", "kita", "kitalah", "kok", "kurang", "lagi", "lagian", "lah", "lain", "lainnya", "lalu", "lama", "lamanya", "lanjut", "lanjutnya", "lebih", "lewat", "lima", "luar", "macam", "maka", "makanya", "makin", "malah", "malahan", "mampu", "mampukah", "mana", "manakala", "manalagi", "masa", "masalah", "masalahnya", "masih", "masihkah", "masing", "masing-masing", "mau", "maupun", "melainkan", "melakukan", "melalui", "melihat", "melihatnya", "memang", "memastikan", "memberi", "memberikan", "membuat", "memerlukan", "memihak", "meminta", "memintakan", "memisalkan", "memperbuat", "mempergunakan", "memperkirakan", "memperlihatkan", "mempersiapkan", "mempersoalkan", "mempertanyakan", "mempunyai", "memulai", "memungkinkan", "menaiki", "menambahkan", "menandaskan", "menanti", "menanti-nanti", "menantikan", "menanya", "menanyai", "menanyakan", "mendapat", "mendapatkan", "mendatang", "mendatangi", "mendatangkan", "menegaskan", "mengakhiri", "mengapa", "mengatakan", "mengatakannya", "mengenai", "mengerjakan", "mengetahui", "menggunakan", "menghendaki", "mengibaratkan", "mengibaratkannya", "mengingat", "mengingatkan", "menginginkan", "mengira", "mengucapkan", "mengucapkannya", "mengungkapkan", "menjadi", "menjawab", "menjelaskan", "menuju", "menunjuk", "menunjuki", "menunjukkan", "menunjuknya", "menurut", "menuturkan", "menyampaikan", "menyangkut", "menyatakan", "menyebutkan", "menyeluruh", "menyiapkan", "merasa", "mereka", "merekalah", "merupakan", "meski", "meskipun", "meyakini", "meyakinkan", "minta", "mirip", "misal", "misalkan", "misalnya", "mula", "mulai", "mulailah", "mulanya", "mungkin", "mungkinkah", "nah", "naik", "namun", "nanti", "nantinya", "nyaris", "nyatanya", "oleh", "olehnya", "pada", "padahal", "padanya", "pak", "paling", "panjang", "pantas", "para", "pasti", "pastilah", "penting", "pentingnya", "per", "percuma", "perlu", "perlukah", "perlunya", "pernah", "persoalan", "pertama", "pertama-tama", "pertanyaan", "pertanyakan", "pihak", "pihaknya", "pukul", "pula", "pun", "punya", "rasa", "rasanya", "rata", "rupanya", "saat", "saatnya", "saja", "sajalah", "saling", "sama", "sama-sama", "sambil", "sampai", "sampai-sampai", "sampaikan", "sana", "sangat", "sangatlah", "satu", "saya", "sayalah", "se", "sebab", "sebabnya", "sebagai", "sebagaimana", "sebagainya", "sebagian", "sebaik", "sebaik-baiknya", "sebaiknya", "sebaliknya", "sebanyak", "sebegini", "sebegitu", "sebelum", "sebelumnya", "sebenarnya", "seberapa", "sebesar", "sebetulnya", "sebisanya", "sebuah", "sebut", "sebutlah", "sebutnya", "secara", "secukupnya", "sedang", "sedangkan", "sedemikian", "sedikit", "sedikitnya", "seenaknya", "segala", "segalanya", "segera", "seharusnya", "sehingga", "seingat", "sejak", "sejauh", "sejenak", "sejumlah", "sekadar", "sekadarnya", "sekali", "sekali-kali", "sekalian", "sekaligus", "sekalipun", "sekarang", "sekarang", "sekecil", "seketika", "sekiranya", "sekitar", "sekitarnya", "sekurang-kurangnya", "sekurangnya", "sela", "selain", "selaku", "selalu", "selama", "selama-lamanya", "selamanya", "selanjutnya", "seluruh", "seluruhnya", "semacam", "semakin", "semampu", "semampunya", "semasa", "semasih", "semata", "semata-mata", "semaunya", "sementara", "semisal", "semisalnya", "sempat", "semua", "semuanya", "semula", "sendiri", "sendirian", "sendirinya", "seolah", "seolah-olah", "seorang", "sepanjang", "sepantasnya", "sepantasnyalah", "seperlunya", "seperti", "sepertinya", "sepihak", "sering", "seringnya", "serta", "serupa", "sesaat", "sesama", "sesampai", "sesegera", "sesekali", "seseorang", "sesuatu", "sesuatunya", "sesudah", "sesudahnya", "setelah", "setempat", "setengah", "seterusnya", "setiap", "setiba", "setibanya", "setidak-tidaknya", "setidaknya", "setinggi", "seusai", "sewaktu", "siap", "siapa", "siapakah", "siapapun", "sini", "sinilah", "soal", "soalnya", "suatu", "sudah", "sudahkah", "sudahlah", "supaya", "tadi", "tadinya", "tahu", "tahun", "tak", "tambah", "tambahnya", "tampak", "tampaknya", "tandas", "tandasnya", "tanpa", "tanya", "tanyakan", "tanyanya", "tapi", "tegas", "tegasnya", "telah", "tempat", "tengah", "tentang", "tentu", "tentulah", "tentunya", "tepat", "terakhir", "terasa", "terbanyak", "terdahulu", "terdapat", "terdiri", "terhadap", "terhadapnya", "teringat", "teringat-ingat", "terjadi", "terjadilah", "terjadinya", "terkira", "terlalu", "terlebih", "terlihat", "termasuk", "ternyata", "tersampaikan", "tersebut", "tersebutlah", "tertentu", "tertuju", "terus", "terutama", "tetap", "tetapi", "tiap", "tiba", "tiba-tiba", "tidak", "tidakkah", "tidaklah", "tiga", "tinggi", "toh", "tunjuk", "turut", "tutur", "tuturnya", "ucap", "ucapnya", "ujar", "ujarnya", "umum", "umumnya", "ungkap", "ungkapnya", "untuk", "usah", "usai", "waduh", "wah", "wahai", "waktu", "waktunya", "walau", "walaupun", "wong", "yaitu", "yakin", "yakni", "yang"]
# stopwords.extend(extender)

class RSummarizer():

    def __init__(self, articles):

        self._articles = []
        for doc in articles:
            with open(DOC_ROOT + doc , encoding="mbcs") as f:
                headline = f.readline()
                url = f.readline()
                f.readline()
                body = f.read().replace('\n', ' ')
                if not self.valid_input(headline, body):
                    self._articles.append((None, None))
                    continue
                self._articles.append((headline, body))


    def valid_input(self, headline, article_text):
        return headline != '' and article_text != ''


    def tokenize_and_stem(self, text):
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
        filtered = []

        # filter out numeric tokens, raw punctuation, etc.
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                filtered.append(token)
        stems = [stemmer.stem(t) for t in filtered]
        return stems


    def score(self, article):
        """ Assigns each sentence in the document a score based on the sum of features values.
            Based on 4 features: relevance to headline, length, sentence position, and TF*IDF frequency.
        """

        headline = article[0]
        sentences = self.split_into_sentences(article[1])
        frequency_scores = self.frequency_scores(article[1])

        for i, s in enumerate(sentences):
            headline_score = self.headline_score(headline, s) * 1.5
            length_score = self.length_score(self.split_into_words(s)) * 1.0
            position_score = self.position_score(float(i+1), len(sentences)) * 1.0
            frequency_score = frequency_scores[i] * 4
            score = (headline_score + frequency_score + length_score + position_score) / 4.0
            self._scores[s] = score


    def generate_summaries(self):
        """ If article is shorter than the desired summary, just return the original articles."""

        # Rare edge case (when total num sentences across all articles is smaller than desired summary length)
        total_num_sentences = 0
        for article in self._articles:
            total_num_sentences += len(self.split_into_sentences(article[1]))

        if total_num_sentences <= SUMMARY_LENGTH:
            return [x[1] for x in self._articles]

        self.build_TFIDF_model()  # only needs to be done once

        self._scores = Counter()
        for article in self._articles:
            self.score(article)

        highest_scoring = self._scores.most_common(SUMMARY_LENGTH)
        if DEBUG:
            print(highest_scoring)

        print("## Headlines: ")
        for article in self._articles:
            print("- " + article[0])

        # Appends highest scoring "representative" sentences, returns as a single summary paragraph.
        return ' '.join([sent[0] for sent in highest_scoring])


    ## ----- STRING PROCESSING HELPER FUNCTIONS -----

    def split_into_words(self, text):
        """ Split a sentence string into an array of words """
        try:
            text = re.sub(r'[^\w ]', '', text) # remove non-words
            return [w.strip('.').lower() for w in text.split()]
        except TypeError:
            return None

    def remove_smart_quotes(self, text):
        """ Only concerned about smart double quotes right now. """
        return text.replace(u"\u201c","").replace(u"\u201d", "")


    def split_into_sentences(self, text):
        tok = nltk.data.load('tokenizers/punkt/english.pickle')
        sentences = tok.tokenize(self.remove_smart_quotes(text))
        sentences = [sent.replace('\n', '') for sent in sentences if len(sent) > 10]
        return sentences


    ## ----- CALCULATING WEIGHTS FOR EACH FEATURE -----

    def headline_score(self, headline, sentence):
        """ Gives sentence a score between (0,1) based on percentage of words common to the headline. """
        title_stems = [stemmer.stem(w) for w in headline if w not in stop_words]
        sentence_stems = [stemmer.stem(w) for w in sentence if w not in stop_words]
        count = 0.0
        for word in sentence_stems:
            if word in title_stems:
                count += 1.0
        score = count / len(title_stems)
        return score


    def length_score(self, sentence):
        """ Gives sentence score between (0,1) based on how close sentence's length is to the ideal length."""
        len_diff = math.fabs(ideal_sent_length - len(sentence))
        return len_diff / ideal_sent_length


    def position_score(self, i, size):
        """ Yields a value between (0,1), corresponding to sentence's position in the article.
            Assuming that sentences at the very beginning and ends of the article have a higher weight. 
            Values borrowed from https://github.com/xiaoxu193/PyTeaser
        """

        relative_position = i / size
        if 0 < relative_position <= 0.1:
            return 0.17
        elif 0.1 < relative_position <= 0.2:
            return 0.23
        elif 0.2 < relative_position <= 0.3:
            return 0.14
        elif 0.3 < relative_position <= 0.4:
            return 0.08
        elif 0.4 < relative_position <= 0.5:
            return 0.05
        elif 0.5 < relative_position <= 0.6:
            return 0.04
        elif 0.6 < relative_position <= 0.7:
            return 0.06
        elif 0.7 < relative_position <= 0.8:
            return 0.04
        elif 0.8 < relative_position <= 0.9:
            return 0.04
        elif 0.9 < relative_position <= 1.0:
            return 0.15
        else:
            return 0


    def build_TFIDF_model(self):
        """ Build term-document matrix containing TF-IDF score for each word in each document
            in the Reuters corpus (via NLTK).
        """
        token_dict = {}
        for article in reuters.fileids():
            token_dict[article] = reuters.raw(article)

        # Use TF-IDF to determine frequency of each word in our article, relative to the
        # word frequency distributions in corpus of 11k Reuters news articles.
        self._tfidf = TfidfVectorizer(tokenizer=self.tokenize_and_stem, stop_words='english', decode_error='ignore')
        tdm = self._tfidf.fit_transform(token_dict.values())  # Term-document matrix


    def frequency_scores(self, article_text):
        """ Individual (stemmed) word weights are then calculated for each
            word in the given article. Sentences are scored as the sum of their TF-IDF word frequencies.
        """

        # Add our document into the model so we can retrieve scores
        response = self._tfidf.transform([article_text])
        feature_names = self._tfidf.get_feature_names() # these are just stemmed words

        word_prob = {}  # TF-IDF individual word probabilities
        for col in response.nonzero()[1]:
            word_prob[feature_names[col]] = response[0, col]
        if DEBUG:
            print(word_prob)

        sent_scores = []
        for sentence in self.split_into_sentences(article_text):
            score = 0
            sent_tokens = self.tokenize_and_stem(sentence)
            for token in (t for t in sent_tokens if t in word_prob):
                score += word_prob[token]

            # Normalize score by length of sentence, since we later factor in sentence length as a feature
            sent_scores.append(score / len(sent_tokens))

        return sent_scores

    # codes #

    #collecting tweet by query
    def getTweet(argv):
        if len(argv) == 0:
            print('You must pass some parameters. Use \"-h\" to help.')
            return

        if len(argv) == 1 and argv[0] == '-h':
            f = open('exporter_help_text.txt', 'r')
            print(f.read())
            f.close()

            return

        try:
            opts, args = getopt.getopt(argv, "", (
            "output=", "username=", "near=", "within=", "since=", "until=", "querysearch=", "toptweets", "maxtweets=", "output=", "lang="))

            tweetCriteria = got.manager.TweetCriteria()
            for opt, arg in opts:
                if opt == '--output':
                    outputFileName = arg
                elif opt == '--username':
                    tweetCriteria.username = arg

                elif opt == '--since':
                    tweetCriteria.since = arg

                elif opt == '--until':
                    tweetCriteria.until = arg

                elif opt == '--querysearch':
                    tweetCriteria.querySearch = arg

                elif opt == '--toptweets':
                    tweetCriteria.topTweets = True

                elif opt == '--maxtweets':
                    tweetCriteria.maxTweets = int(arg)

                elif opt == '--near':
                    tweetCriteria.near = '"' + arg + '"'

                elif opt == '--within':
                    tweetCriteria.within = '"' + arg + '"'

                elif opt == '--output':
                    outputFileName = arg

                elif opt == '--lang':
                    tweetCriteria.lang = arg

            #membuat file csv
            outputFile = csv.writer(open(outputFileName, "w", encoding='utf-8-sig', newline=''), delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)

            outputFile.writerow(
                ['username','date','retweets','favorites','text','geo','mentions','hashtags','id','permalink', 'emoji'])

            print('Collecting tweets...\n')

            def receiveBuffer(tweets):
                for t in tweets:
                    add_list = []
                    if (isinstance(t.emojis, list)):
                        emoji = ' '.join(t.emojis)
                    else:
                        emoji = t.emojis
                    for each in [t.username, t.date.strftime("%Y-%m-%d %H:%M"), t.retweets, t.favorites, t.text, t.geo, t.mentions, t.hashtags, t.id, t.permalink, emoji]:
                        add_list.append(each)
                    #mendapatkan tweet sesuai query dan memasukkan ke csv
                    outputFile.writerow(add_list)
                print('%d tweets saved on file...\n' % len(tweets))

            got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)

        except :
            print('Arguments parse error, failed collecting tweets')
        finally:
            print('Done. Output file generated "%s".' % outputFileName)


    def trainAnago(train,valid,test):
        x_train, y_train = load_data_and_labels(train)
        x_valid, y_valid = load_data_and_labels(valid)
        x_test, y_test = load_data_and_labels(test)
        # karena hasil tdk konsisten, random seednya diisi manual
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(42)
        rn.seed(12345)

        # mengatur parameter
        model = anago.Sequence(char_emb_size=25, word_emb_size=100, char_lstm_units=25,
                               word_lstm_units=100, dropout=0.5, char_feature=True, crf=True,
                               batch_size=20, optimizer='adam', learning_rate=0.001, lr_decay=0.9,
                               clip_gradients=5.0, max_epoch=35, early_stopping=True, patience=3, train_embeddings=True,
                               max_checkpoints_to_keep=5, log_dir=None)
        model.train(x_train, y_train, x_valid, y_valid)
        print("\n\nEvaluasi Test:")
        model.eval(x_test, y_test)
        return model


    def normalize_repeated_char(tokenized_sentence):
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 
                    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 
                    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        tokenized_sentence = ' '.join(tokenized_sentence)
        for i in range(len(alphabet)):
            charac_long = 5
            while charac_long >= 2:
                char = alphabet[i] * charac_long
                tokenized_sentence = tokenized_sentence.replace(char, alphabet[i])
                charac_long -= 1        
        tokenized_sentence = tokenized_sentence.split()    
        return tokenized_sentence


    def delete_emoji(sentence):
        emoji_pattern = re.compile("["
            u"\U0001F600-\U0001F64F"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
        for word in sentence:
            for letter in word:
                print(letter)
    #             letter = emoji_pattern.sub(r'', letter)
    #     deleted_emoji = emoji_pattern.sub(r'', ContentNoSlang)
    #     return deleted_emoji


    def deEmojify(inputString):
        return inputString.encode('ascii', 'ignore').decode('ascii')


    def preprocessing_tweets_first(self, text):
        clean = html.unescape(text) #Handling Unicode HTML
        clean = re.sub("[0-9]", "", clean) #RemoveNumber
        clean = re.sub(r"http\S+", "", clean) #Remove links
        clean = self.deEmojify(clean) #Remove Emoji
        clean = " ".join(re.findall("[#a-zA-Z]{3,}", clean)) #Remove Puntc
        content = list()
        content.append(clean)
        return " ".join(content)


    def stopword(words):
        clean = []
        for word in words:
            if word not in stopwords:
                clean.append(word)
        return (" ".join(clean))


    def normalize_slang_words(text):
        slang_word_dict = json.loads(open('slang_word_dict.txt', 'r').read())
        words = text.split()
        for index in range(len(words)):
            for key, value in slang_word_dict.items():
                for v in value:
                    if words[index] == v:
                        words[index] = key
                    else:
                        continue 
        content = list()
        content.append(words)

        return " ".join(content)


    def preprocessing_tweets_second(self,text):
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        clean   = stemmer.stem(text) 
        clean = re.sub("#", "", clean)
        clean = stopword(clean.split()) #Stopword
        clean
    #     clean = normalize_slang_words(clean)
        return clean


    def preprocessing_tweets_clustering(self,text):
        clean = html.unescape(text) #Handling Unicode HTML
        clean = re.sub("[0-9]", "", clean) #RemoveNumber
        clean = re.sub(r"http\S+", "", clean) #Remove links
        clean = re.sub(r"pic.twitter.com\S+", "", clean) #Remove picts
        clean = self.deEmojify(clean) #Remove Emoji
        clean = " ".join(re.findall("[#a-zA-Z]{3,}", clean)) #Remove Puntc
        content = list()
        content.append(clean)
        x = " ".join(content)
        clean = x
        clean = re.sub("#", "", clean)
        clean = self.stopword(clean.split()) #Stopword
        # clean = self.normalize_slang_words(clean)
        return clean


    def kMedoids(D, k, tmax=100):
        # determine dimensions of distance matrix D
        m, n = D.shape

        if k > n:
            raise Exception('too many medoids')

        # find a set of valid initial cluster medoid indices since we
        # can't seed different clusters with two points at the same location
        valid_medoid_inds = set(range(n))
        invalid_medoid_inds = set([])
        rs,cs = np.where(D==0)
        # the rows, cols must be shuffled because we will keep the first duplicate below
        index_shuf = list(range(len(rs)))
        np.random.shuffle(index_shuf)
        rs = rs[index_shuf]
        cs = cs[index_shuf]
        for r,c in zip(rs,cs):
            # if there are two points with a distance of 0...
            # keep the first one for cluster init
            if r < c and r not in invalid_medoid_inds:
                invalid_medoid_inds.add(c)
        valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

        if k > len(valid_medoid_inds):
            raise Exception('too many medoids (after removing {} duplicate points)'.format(
                len(invalid_medoid_inds)))

        # randomly initialize an array of k medoid indices
        M = np.array(valid_medoid_inds)
        np.random.shuffle(M)
        M = np.sort(M[:k])

        # create a copy of the array of medoid indices
        Mnew = np.copy(M)

        # initialize a dictionary to represent clusters
        C = {}
        for t in range(tmax):
            # determine clusters, i. e. arrays of data indices
            J = np.argmin(D[:,M], axis=1)
            for kappa in range(k):
                C[kappa] = np.where(J==kappa)[0]
            # update cluster medoids
            for kappa in range(k):
                J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
                j = np.argmin(J)
                Mnew[kappa] = C[kappa][j]
            np.sort(Mnew)
            # check for convergence
            if np.array_equal(M, Mnew):
                break
            M = np.copy(Mnew)
        else:
            # final update of cluster memberships
            J = np.argmin(D[:,M], axis=1)
            for kappa in range(k):
                C[kappa] = np.where(J==kappa)[0]

        # return results
        return M, C


# TODO
    def test_args_kwargs(arg1, arg2, arg3):
        print("arg1:", arg1)
        print("arg2:", arg2)
        print("arg3:", arg3)

 
    def test():
        print("tester")


    def remove_links():
        return
'''
. filtrasi tweets
. menganalisan dan melabeli
. 

'''