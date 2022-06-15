import joblib
from nltk import word_tokenize
from numpy.core.defchararray import lower
from sklearn.pipeline import Pipeline

#Сюда добавлять новые тригеры для отписки
bad_words = [
    'ебаная',
    'ебаный',
    'шлюха',
    'шлюхи',
    'отпиши',
    'не пишите',
    'ебанутые',
    'хуй',
    'хер',
    'сука',
    'суки',
    'отпишите',
    'не пиши',
    'ебал',
    'хуесос',
    'пизда',
    'тварь',
    'твари',
    'шлюх',
    'мрази',
    'выбляд',
    'гандон',
    'говноед',
    'уебки',
    'уебок',
    'пидорас',
    'уебище',
    'заебали',
    'заебешь',
    'отъебитесь',
    'отьебитесь',
    'прошманда',
    'шкура',
    'пасосник',
    'долбоеб',
    'далбаеб',
    'долбан',
    'блядина',
    'проститутка',
    'кончненые',
    'конч',
    'конченные',
    'ненормальные',
    'ебнутые',
    'припизднутые',
    'шлюхи ',
    'хуесосы',
    'уебаки',
    'уебок',
    'уеище',
    'не пишите',
    'хули',
    'сучка',
    'рот ебал ',
    'прошмандовка ',
    'хуесосина',
    'нахуй',
    'пидорасы',
    'гандоны ',
    'пидорасы',
    'ахуели',
    'ахуел',
    'отписка']

loaded_model = joblib.load('/Users/uhome/Desktop/MODEL_CLASSIFIER/finalized_model.sav')
loaded_vectorizer = joblib.load('/Users/uhome/Desktop/MODEL_CLASSIFIER/vectorizer.sav')
model_pipeline_c_12 = Pipeline([
    ("vectorizer", loaded_vectorizer),
    ("model", loaded_model)
])


def ask(text):
    mat = False
    for object in lower(word_tokenize(text)):
        if object in bad_words:
            mat = True
            break
    if mat:
        return 0
    else:
        return model_pipeline_c_12.predict([text])


# Пример работы классификатора
def classifier(text):
    if ask(text) == 1:
        print("Я понял, больше не побеспокою")
    elif ask(text) == 2:
        print("Сейчас разъясню подробнее")
    elif ask(text) == 3:
        print("Тогда скоро будем у вас, ожидайте!")
    else: print("Отписали вас от рассылки!")

text = "Отпишите меня"
print(text)
classifier(text)













