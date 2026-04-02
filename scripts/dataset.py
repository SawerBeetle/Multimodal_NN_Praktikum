import os
import sys

import albumentations as A
from albumentations.pytorch import ToTensorV2
import joblib
import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import Dataset, Subset
from torchvision.datasets import ImageFolder
import yaml

"""  
Импортируем конфиг и данные. 
"""
path_to_config = os.path.join(os.getcwd(), 'config', 'config.yaml')

with open(path_to_config, "r") as f:
    config_notebook = yaml.safe_load(f)

ingredients_path = os.path.join(os.getcwd(), 'data', 'ingredients.csv')
# загрузим таблицу ингредиентов
ingredients = pd.read_csv(ingredients_path)

dishes_train = joblib.load('imports/dishes_train.pkl')
dishes_val = joblib.load('imports/dishes_val.pkl')
dishes_test = joblib.load('imports/dishes_test.pkl')
images_train = joblib.load('imports/images_train.pkl')
images_val = joblib.load('imports/images_val.pkl')
images_test = joblib.load('imports/images_test.pkl')

images_path = os.path.join(os.getcwd(), 'data', 'images')
images = ImageFolder(images_path)

"""Если код отлаживается, то для ускорения работы на локальной машине 
уменьшим объём данных. 
"""
if config_notebook['mode'] == 'preliminar': 
    dishes_train = dishes_train.iloc[0:16]
    images_train = Subset(images, dishes_train['dish_id'])
    dishes_val = dishes_val.iloc[0:16]
    images_val = Subset(images, dishes_val['dish_id'])
    dishes_test = dishes_test.iloc[0:16]
    images_test = Subset(images, dishes_test['dish_id'])
    print('Код выполняется в режиме отладки, объём данных уменьшен до 16 объектов.')

"""Если код отлаживается, то для ускорения работы на локальной машине 
уменьшим объём батча, для завершающего этапа работы на ВМ выставим нормальный. 
Значение MAE, при котором завершаем обучение, наоборот, увеличим.
"""
if config_notebook['mode'] == 'preliminar': 
    BATCH_SIZE = 8
    VAL_MAE = 180
else: 
    BATCH_SIZE = 64
    VAL_MAE = 50

""" 
Заменим словами ID ингредиентов во фреймах 'dishes_train' 
и 'dishes_test'. Для этого построим словарь на основе фрейма 
'ingredients' и определим функцию, позволяющую применять его 
автоматически. Исходим из того, что значимая информация в ID 
заключена в трёх последних цифрах. 

Здесь же рассчитаем количество калорий на грамм – будущую 
целевую переменную. 
"""
# для замены ID ингредиентов на их названия создаём словарь
ingredients_dict = dict(zip(ingredients['id'], ingredients['ingr']))

# определяем функцию для замены ID на названия
def id_to_ingr(data): 
    # разделяем ID по ';'
    data = data.split(';')
    # берём три последних символа в ID
    data = [ingr[-3:] for ingr in data]

    # создаём пустой список для названий
    ingr_text = []
    # Меняем ID на название, 
    # добавляем название в 'ingr_text'. 
    for ingr in data: 
        ingr_text.append(ingredients_dict[int(ingr)])

    return(ingr_text)


# map должно сберечь сколько-то ресурсов по сравнению с циклом
dishes_train['ingredients'] = dishes_train['ingredients'].map(id_to_ingr)
dishes_val['ingredients'] = dishes_val['ingredients'].map(id_to_ingr)
dishes_test['ingredients'] = dishes_test['ingredients'].map(id_to_ingr)

"""  
Исходный формат данных и ранее проведённая обработка подразумевают, 
что все данные, кроме изображений, подаются в формате pandas data frame, 
изображения в формате Subset, откуда они уже будут извлекаться. 
"""
class MultimodalDataset(Dataset):
    def __init__(self, df, images, mode='train'):
        super().__init__()
        # get data frame
        self.df = df
        # изображения
        self.images = images
        # текстовая модель
        # self.text_model = config_notebook['text_model']
        # конфиги модели для обработки фото
        self.image_cfg = timm.get_pretrained_cfg(config_notebook['image_model'])
        # токенизатор
        # self.tokenizer = AutoTokenizer.from_pretrained(config_notebook['text_model'])
        # аугментатор
        if mode == 'train': 
            self.augmentator = A.Compose([
                A.OneOf([
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.2, 0.2),
                        contrast_limit=(-0.2, 0.2),
                        p=0.15
                    ), 
                    A.HueSaturationValue(
                        hue_shift_limit=(-20, 20), 
                        sat_shift_limit=(-30, 30), 
                        val_shift_limit=(-20, 20), 
                        p=0.20
                    ),
                    A.ToGray(p=0.03)
                ]), 
                A.CoarseDropout(
                    num_holes_range=(1, 8), 
                    hole_height_range=(.05, 0.15), 
                    hole_width_range=(0.05, 0.15), 
                    fill='random_uniform', 
                    p=0.2
                ), 
                A.Resize(
                    height=self.image_cfg.input_size[1], 
                    width=self.image_cfg.input_size[2]
                    ), 
                A.Normalize(self.image_cfg.mean, self.image_cfg.std), 
                ToTensorV2()
            ], seed=config_notebook['seed'])
        else:
            self.augmentator = A.Compose([
                A.Resize(
                    height=self.image_cfg.input_size[1], 
                    width=self.image_cfg.input_size[2]
                    ), 
                A.Normalize(self.image_cfg.mean, self.image_cfg.std), 
                ToTensorV2()
            ], seed=config_notebook['seed'])
            
    def __len__(self): 
        num_rows = len(self.df)
        num_images = len(self.images)
        
        # сравним количество объектов во фрейме и наборе фото
        if num_rows != num_images: 
            print('Количество изображений и описаний не совпадают.')
            sys.exit()
        else: 
            return num_rows

    def __getitem__(self, idx):
        # получить и аугментировать изображение (и сразу достать его код)
        """  
        Эту невменяемую конструкцию я получил с помощью гугловской нейросетки. 
        Мой код обрабатывал images то как ImageFolder, то как Subset. 
        Где была ошибка – я потом нашёл, но проще пока оставить, как есть. 
        """
        # get image name
        row = self.df.iloc[idx]
        image_id = row['dish_id']
        # извлечь ImageFolder из Subset
        full_dataset = self.images
        # если не извлеклось (Subset внутри Subset), повторять до результата
        while hasattr(full_dataset, 'dataset'):
            full_dataset = full_dataset.dataset
        # извлечь классы
        class_to_idx = full_dataset.class_to_idx
        # выбрать нужный (соответствующий ID блюда)
        class_idx = class_to_idx[image_id]
        # вытащить изображение с соответстствующим классом
        orig_idx = full_dataset.targets.index(class_idx)
        image_pil, _ = full_dataset[orig_idx]

        # преобразовать его в массив numpy
        image_np = np.array(image_pil)
        # аугментировать изображение
        augmented = self.augmentator(image=image_np)
        image = augmented['image'] 

        # get calories and mass
        calories = row['total_calories']
        mass = row['total_mass']

        return{
            'id': image_id, 
            'image': image, 
            'calories': calories, 
            'mass': mass
        }

# создадим датасеты и сохраним их, чтобы избежать проблем с импортом в другой скрипт
ds_train = MultimodalDataset(dishes_train, images_train)
joblib.dump(ds_train, 'imports/ds_train.pkl')
ds_val = MultimodalDataset(dishes_val, images_val, mode='test')
joblib.dump(ds_val, 'imports/ds_val.pkl')
ds_test = MultimodalDataset(dishes_test, images_test, mode='test')
joblib.dump(ds_test, 'imports/ds_test.pkl')

