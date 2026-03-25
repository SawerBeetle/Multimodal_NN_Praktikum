import os

from functools import partial
import itertools
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader #, Subset
from tqdm.auto import tqdm
import yaml

"""  
Импортируем конфиг и данные. 
"""
path_to_config = os.path.join(os.getcwd(), 'config', 'config.yaml')

with open(path_to_config, "r") as f:
    config_notebook = yaml.safe_load(f)

# ingredients_path = os.path.join(os.getcwd(), 'data', 'ingredients.csv')
# # загрузим таблицу ингредиентов
# ingredients = pd.read_csv(ingredients_path)

dishes_test = joblib.load('imports/dishes_test.pkl')
ds_test = joblib.load('imports/ds_test.pkl')
images_test = joblib.load('imports/images_test.pkl')

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
Обеспечим воспроизводимость. 

Если я правильно понял, то manual_seed устанавливает генератор 
случайных чисел для cpu, cuda.manual_seed для cuda, а torch.Generator 
нужен отдельно для DataLoader'ов. В них, кроме того, установлен seed 
для worker_init_fn, но зачем это делать отдельно, я толком не понял, 
и не уверен, нужно ли вообще. 

Включать детерминированные алгоритмы для полной воспроизводимости я не стал, 
так как источники дружно пугают резким снижением скорости вычислений из-за этого. 
"""
torch.manual_seed(config_notebook['seed'])
if torch.cuda.is_available():
        torch.cuda.manual_seed(config_notebook['seed'])
        torch.cuda.manual_seed_all(config_notebook['seed'])
g = torch.Generator()
g.manual_seed(config_notebook['seed']);

"""  
Определим collate_fn. 
"""
def collate_fn(batch):
    # код блюда
    id = [item['id'] for item in batch]
    # фото
    images = torch.stack([item['image'] for item in batch])
    # токенизированные названия ингредиентов и маска
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # калорийность (абсолютная и на грамм веса)
    calories = torch.FloatTensor([item['calories'] for item in batch])
    calories_per_g = torch.FloatTensor([item['calories_per_g'] for item in batch])
    # масса блюда
    mass = torch.FloatTensor([item['mass'] for item in batch])

    return{
        'id': id, 
        'image': images, 
        'input_ids': input_ids, 
        'attention_mask': attention_mask, 
        'calories': calories, 
        'calories_per_g': calories_per_g, 
        'mass': mass
    }    

loader_test = DataLoader(
    ds_test, 
    batch_size=BATCH_SIZE, 
    shuffle=False, 
    collate_fn=partial(collate_fn), 
    worker_init_fn=np.random.seed(config_notebook['seed']), 
    generator=g
    )

"""  
Определим функцию для тестирования мультимодальной модели. 
Помимо MAE, она должна выдать ещё и пять объектов с наихудшим прогнозом. 
"""
# определим код для тестирования
def model_test(model, test_loader, images):
    # определить, с каким устройством (cpu/cuda) работаем
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # переводим модель в режим инференса
    model.eval()
    # инициализируем пустые списки
    # абсолютные ошибки
    all_absolute_errors = []
    # ID
    all_ids = []
    # реальная калорийность
    all_calories = []
    # предсказанная калорийность
    all_predicted_calories = []
    
    # отключаем расчёт градиентов
    with torch.no_grad():
        # проходим по батчам
        for _, batch in enumerate(
            tqdm(test_loader, total=len(test_loader), leave=False, desc="test")
            ): 
            # входные данные для работы моделей
            inputs = {
                'input_ids': batch['input_ids'].to(DEVICE), 
                'attention_mask': batch['attention_mask'].to(DEVICE), 
                'image': batch['image'].to(DEVICE)
            }
            # переменные для расчёта MAE
            calories = batch['calories'].to(DEVICE)
            mass = batch['mass'].to(DEVICE)
            # ID объектов
            ids = batch['dish_id'].to(DEVICE)

            # получим результат работы модели (калорийность на грамм)
            result = model(**inputs)
            # рассчитаем калорийность блюда целиком
            results_total = result * mass
            # рассчитаем абсолютные отклонения
            absolute_errors = torch.abs(results_total - calories)

            """  
            Здесь добавим данные по набору в целом к соответствующим спискам, 
            а потом преобразуем его во фрейм, который отсортируем и выдадим 
            наружу. 
            """
            # ID
            all_ids.extend(ids.tolist())
            # реальная калорийность
            all_calories.extend(calories.tolist())
            # предсказанная калорийность
            all_predicted_calories.extend(results_total.tolist())
            # абсолютные ошибки 
            all_absolute_errors.extend(absolute_errors.tolist())
        
        # преобразуем списки так, чтобы не было вложений
        # ID
        all_ids = list(itertools.chain.from_iterable(all_ids))
        # реальная калорийность
        all_calories = list(itertools.chain.from_iterable(all_calories))
        # предсказанная калорийность
        all_predicted_calories = list(itertools.chain.from_iterable(all_predicted_calories))
        # абсолютные ошибки
        all_absolute_errors = list(itertools.chain.from_iterable(all_absolute_errors))
        # создадим фрейм 
        results_df = pd.DataFrame({
            'dish_id': all_ids, 
            'total_calories': all_calories, 
            'predicted_calories': all_predicted_calories, 
            'absolute_error': all_absolute_errors
        })
        # отсортируем фрейм по величине ошибки
        results_df = results_df.sort_values(by='absolute_error', ascending=False)

        # рассчитываем MAE для тестового набора
        test_mae = sum(all_absolute_errors) / len(all_absolute_errors)

    print(f"Значение MAE для тестового набора равно {test_mae:.4f}. ")
    print('-' * 15)
    print('Блюда с наихудшим предсказанием калорийности: ')

    # вызовем индексы этих блюд в 'images', соответствующие 'dish_id'
    worst_predicted = []
    for dish_id in results_df['dish_id']: 
        worst_predicted.append(images.class_to_idx[dish_id])

    # картинки на экран
    fig = plt.figure(figsize=(10, 20))
    index = 1
    for idx in worst_predicted: 
        img, label = images[idx]
        plt.subplot(1, 5, index)
        plt.imshow(img)
        plt.title(images.classes[label])    
        plt.axis('off')
        index += 1
    plt.tight_layout() 
    plt.show()
    
    return results_df
