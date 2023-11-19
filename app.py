from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
import io

# model_fileName = 'model100_all.h5'
@st.cache_resource#
def load__model(model):
   return load_model(model)

model = load__model('model100_all.h5')
img_width, img_height = 32, 32

classes = ['яблоко', 'аквариумная рыба', 'ребенок', 'медведь', 'бобёр', # 1-5
            'кровать', 'пчела', 'жук', 'велосипед', 'бутылка', # 6-10
            'миска', 'мальчик', 'мост', 'автобус', 'бабочка', # 11-15
            'верблюд', 'банка', 'замок', 'гусеница', 'скот', # 16-20
            'стул', 'шимпанзе', 'часы', 'облако', 'таракан', # 21-25
            'диван', 'краб', 'крокодил', 'чашка', 'динозавр', # 26-30
            'дельфин', 'слон', 'камбала', 'лес', 'лиса', # 31-35
            'девочка', 'хомяк', 'дом', 'кенгуру', 'комп. клавиатура', # 36-40
            'лампа', 'газонокосилка', 'леопард', 'лев', 'ящерица', # 41-45
            'лобстер', 'человек', 'кленовое дерево', 'мотоцикл', 'гора', # 46-50
            'мышь', 'гриб', 'дуб', 'апельсин', 'орхидея', # 51-55
            'выдра', 'пальма', 'груша', 'грузовик пикап', 'сосна', # 56-60
            'равнина', 'тарелка', 'мак', 'дикобраз', 'опоссум', # 61-65
            'кролик', 'енот', 'скат', 'дорога', 'ракета', # 66-70
            'роза', 'море', 'тюлень', 'акула', 'землеройка', # 71-75
            'скунс', 'небоскреб', 'улитка', 'змея', 'паук', # 76-80
            'белка', 'трамвай', 'подсолнух', 'сладкий перец', 'стол', # 81-85
            'танк', 'телефон', 'телевизор', 'тигр', 'трактор', # 86-90
            'поезд', 'форель', 'тюльпан', 'черепаха', 'шкаф', # 91-95
            'кит', 'ива', 'волк', 'женщина', 'червяк'] # 96-100


# Сделаем все названия классов с заглавной буквы
classes = [classes.capitalize() for classes in classes]


# Распознавание изображения нейросетью

#@st.cache_resource#(allow_output_mutation=True)
def pred_image(model, image):

  pred = model.predict(image)

  dict = {}
  result_other = ''

  for i, cl in enumerate(classes):
    value = int(pred[0, i]*10000)
    if  value >= 100:
      dict[value] = classes[i]

  list_val = sorted(list(dict), reverse=True)

  for i in list_val[1:]:
      result_other +=  '{:<14s} - {:6.2%}\n\n'.format(dict[i], i/10000)

  result = '{:<14s} - {:6.2%}'.format(dict[list_val[0]], list_val[0]/10000)

  # Вычисление индекса класса с максимальным значением вероятности и вывод имени класса
  cls_image = np.argmax(pred)

  return result_other, result#classes[cls_image]

# @st.cache_data
def fill_sidebar():

  sorted_classes = sorted(classes.copy())
  with st.sidebar:
    
    st.markdown("<h3 style='text-align: center; line-height: 1%; color: darkblue;'>Список распознаваемых классов</h3>", 
                unsafe_allow_html=True)
    st.divider()

    column1, column2 = st.columns(2)

    with column1:
        for val in sorted_classes[:50]:
            st.markdown(f"<h5 style='text-align: left; line-height: 1%; color: black;'>{val}</h5>", unsafe_allow_html=True)


    with column2:
        for val in sorted_classes[50:]:
            st.markdown(f"<h5 style='text-align: left; line-height: 1%; color: black;'>{val}</h5>", unsafe_allow_html=True)



fill_sidebar()

st.markdown("<h2 style='text-align: center; line-height: 1%; color: grey;'>Классификация изображений</h2>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center; line-height: 70%; color: darkblue;'>100 классов</h4>", unsafe_allow_html=True)
st.markdown('\n\n\n')


uploaded_file = st.file_uploader(label='Выберите изображение для распознавания')

if uploaded_file is not None:
    image = uploaded_file.getvalue()
    st.image(image)
    image = Image.open(io.BytesIO(image))


   

    result_ = st.button('Распознать изображение')
    if result_:
        img = image.resize((img_width, img_height))


    # Открытие картинки и изменение ее размера для соответсвия входу модели
    #img = Image.open('20.jpg').resize((img_width, img_height))

    # Проверка результата
    #plt.imshow(img)
    #plt.show()

      # Преобразование картинки в numpy-массив чисел с плавающей запятой и нормализация значений пикселей
        image = np.array(img, dtype='float64') / 255.

      # добавление оси для совпадения формы входа модели; получается батч из одного примера
        image = np.expand_dims(image, axis=0)

        # st.write('Результат распознавания:')

        other, res = pred_image(model, image)

        # st.write(other)
        # st.write('Изображен(а): ', res)

        column1, column2 = st.columns(2)

        with column1:
            # st.subheader("Изображен(а)")
            st.markdown("<h4 style='text-align: left; line-height: 50%; color: black;'>Изображен(а)</h4>", 
                        unsafe_allow_html=True,
                        help='Модель обучалась на картинках размером 32х32, \n\n поэтому результаты могут Вас удивить.')
            st.markdown(f"<h5 style='text-align: left; line-height: 100%; color: green;'>{res}</h5>", unsafe_allow_html=True)
            # st.write(f':red[**{res}**]')

        with column2:
            # st.subheader("Остальные результаты (>1%):")
            st.markdown("<h4 style='text-align: left; line-height: 50%; color: black;'>Остальные результаты (>1%):</h4>", unsafe_allow_html=True)
            st.write(other)
