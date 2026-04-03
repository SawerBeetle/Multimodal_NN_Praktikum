import os

from functools import partial
import itertools
import joblib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import yaml

from dataset import MultimodalDataset
from utils import MultimodalModel

if __name__ == "__main__":

    """  
    Импортируем конфиг и данные. 
    """
    path_to_config = os.path.join(os.getcwd(), 'config', 'config.yaml')

    with open(path_to_config, "r") as f:
        config_notebook = yaml.safe_load(f)

    dishes_test = joblib.load('imports/dishes_test.pkl')
    ds_test = joblib.load('imports/ds_test.pkl')
    images_test = joblib.load('imports/images_test.pkl')

    images_path = os.path.join(os.getcwd(), 'data', 'images')
    # загрузим фото
    images_base = ImageFolder(images_path)

    ingredients_path = os.path.join(os.getcwd(), 'data', 'ingredients.csv')
    ingredients = pd.read_csv(ingredients_path)
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
        # меняем ID на название, добавляем название в 'ingr_text'
        for ingr in data: 
            ingr_text.append(ingredients_dict[int(ingr)])

        return(ingr_text)
    
    dishes_test['ingredients'] = dishes_test['ingredients'].map(id_to_ingr)


    """  
    Загрузим веса модели.
    """
    trained_model = MultimodalModel(config_notebook)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.to(device)

    weights_path = os.path.join(os.getcwd(), 'model/' "weights.pth")
    weights = torch.load(weights_path) 

    # Если сохраняли через trainer или просто state_dict
    if 'state_dict' in weights:
        trained_model.load_state_dict(weights['state_dict'], strict=False)
    else:
        trained_model.load_state_dict(weights, strict=False)

    for param in trained_model.parameters():
        param.requires_grad = False
    trained_model.eval()

    """Если код отлаживается, то для ускорения работы на локальной машине 
    уменьшим объём батча, для завершающего этапа работы на ВМ выставим нормальный. 
    Значение MAE, при котором завершаем обучение, наоборот, увеличим.
    """
    if config_notebook['mode'] == 'preliminar': 
        BATCH_SIZE = 8
        VAL_MAE = 250
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
        # калорийность 
        calories = torch.FloatTensor([item['calories'] for item in batch])
        # масса блюда
        mass = torch.FloatTensor([item['mass'] for item in batch])

        return{
            'id': id, 
            'image': images, 
            'calories': calories, 
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
    def test_multimodal_model(model, test_loader, images):
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
                tqdm(
                    test_loader, 
                    total=len(test_loader), 
                    leave=False, 
                    position=0, 
                    desc="test")
                ): 
                # входные данные для работы моделей
                inputs = batch['image'].to(DEVICE)
                # переменные для расчёта MAE
                calories = batch['calories'].to(DEVICE)
                mass = batch['mass'].to(DEVICE)
                # ID объектов
                ids = batch['id']

                # получим результат работы модели (калорийность на грамм)
                result = model(inputs).squeeze()
                # print(len(result))
                # рассчитаем абсолютные отклонения
                absolute_errors = torch.abs(result - calories)

                """  
                Здесь добавим данные по набору в целом к соответствующим спискам, 
                а потом преобразуем его во фрейм, который отсортируем и выдадим 
                наружу. 
                """
                # ID
                all_ids.extend(ids)
                # реальная калорийность
                all_calories.extend(calories.tolist())
                # предсказанная калорийность
                all_predicted_calories.extend(result.tolist())
                # абсолютные ошибки 
                all_absolute_errors.extend(absolute_errors.tolist())
            
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
            test_mae = np.mean(all_absolute_errors)

        results_worst = results_df.head()
        print(f"Значение MAE для тестового набора равно {test_mae:.4f}. ")
        print('-' * 15)
        print('Блюда с наихудшим предсказанием калорийности: ')
        display(results_worst)
        
        return results_worst['dish_id']
    
    worst_classes = test_multimodal_model(
        model=trained_model, test_loader=loader_test, images=images_test
        )

    worst_classes = list(worst_classes)
    worst_classes_idx = []
    for id in worst_classes: 
        worst_classes_idx.append(images_base.class_to_idx[id])

    # картинки на экран
    fig = plt.figure(figsize=(10, 20))
    index = 1
    for idx in worst_classes_idx: 
        img, label = images_base[idx]
        plt.subplot(1, 5, index)
        plt.imshow(img)
        plt.title(worst_classes[index - 1])    
        plt.axis('off')
        index += 1
    plt.tight_layout() 
    plt.show();

    for bad_class in worst_classes:
        print(bad_class, ': ingredients')
        print(
            dishes_test.loc[dishes_test['dish_id'] == bad_class, 'ingredients'].tolist()
            )
        print('-' * 15)
