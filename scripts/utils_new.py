import os
import sys

# from beeply import notes
from functools import partial
import itertools
import joblib
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader #, Subset
from tqdm.auto import tqdm
# from transformers import AutoModel
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

ds_train = joblib.load('imports/ds_train.pkl')
ds_val = joblib.load('imports/ds_val.pkl')

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
    # input_ids = torch.stack([item['input_ids'] for item in batch])
    # attention_mask = torch.stack([item['attention_mask'] for item in batch])
    # калорийность (абсолютная и на грамм веса)
    calories = torch.FloatTensor([item['calories'] for item in batch])
    # calories_per_g = torch.FloatTensor([item['calories_per_g'] for item in batch])
    # масса блюда
    mass = torch.FloatTensor([item['mass'] for item in batch])

    return{
        'id': id, 
        'image': images, 
        # 'input_ids': input_ids, 
        # 'attention_mask': attention_mask, 
        'calories': calories, 
        # 'calories_per_g': calories_per_g, 
        'mass': mass
    }   

"""  
Модель должна обеспечиват обработку текстовых данных 
(список ингредиентов) и картинок. 

Предполагается использование bert-base-uncased и tf_efficientnet_b0. 
Названия моделей прописываются в конфиге, а вот разморозка последних 
слоёв этих моделей прописана в __init__ и заточена именно под них. 
Соответственно, если будут изменены модели в конфиге, 
то код заморозки/разморозки придётся переписывать вручную. 
"""
class MultimodalModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.text_model = AutoModel.from_pretrained(config['text_model']) 
        self.image_model = timm.create_model(
            config['image_model'],
            pretrained=True,
            num_classes=0 
        )
        
        # заморозка весов текстовой модели
        # for param in self.text_model.parameters(): 
        #     param.requires_grad = False
        # разморозка последнего блока текстовой модели
        # for param in self.text_model.pooler.parameters(): 
        #     if config['text_model'] == 'bert-base-uncased': 
        #         param.requires_grad = True
            # если модель не bert_base_uncased, то превать выполнение и изменить код
            # else: 
            #     print('Измените размораживаемые слои текстовой модели. ')
                # прерывание выполнения кода для изменения размораживаемых слоёв
                # sys.exit()    

        # заморозка весов модели для изображений
        for param in self.image_model.parameters(): 
            param.requires_grad = False 
        # разморозка последнего свёрточного блока модели для изображений
        # закомментированное под tf_efficient_b0, раскомментированное под resnet50
        # for param in self.image_model.conv_head.parameters(): 
        for param in self.image_model.fc.parameters(): 
            if config['image_model'] == 'tf_efficientnet_b0': 
                param.requires_grad = True
            # если модель не resnet50, то превать выполнение и изменить код
            else: 
                print('Измените размораживаемые слои модели для изображений. ')
                # прерывание выполнения кода для изменения размораживаемых слоёв
                sys.exit()    

        # приведение выходов обеих моделей к одной размерности
        # self.text_proj = nn.Linear(self.text_model.config.hidden_size, 256)
        self.image_proj = nn.Linear(self.image_model.num_features, 256)

        # определить регрессор
        self.regressor = nn.Sequential(
            # nn.Linear(2 * config['hidden_dim'], config['hidden_dim']),      
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),      
            # nn.LayerNorm(config['hidden_dim']),
            nn.BatchNorm1d(config['hidden_dim'] // 2),         
            nn.ReLU(),                           
            nn.Dropout(0.15),                    
            nn.Linear(config['hidden_dim'] // 2, 1) 
        )

    # прямой проход
    def forward(self, input_ids, attention_mask, image):
        # работа моделей
        # text_features = self.text_model(input_ids, attention_mask).last_hidden_state[:,  0, :]
        image_features = self.image_model(image)

        # приведение результатов работы моделей к общему формату
        # text_emb = self.text_proj(text_features)
        image_emb = self.image_proj(image_features)

        # print(f"Text shape: {text_emb.shape}")   # Ожидаем [8, 256]
        # print(f"Image shape: {image_emb.shape}") # Ожидаем [8, 256]
        # объединение результатов работы моделей
        # fused_emb = torch.cat((text_emb, image_emb), dim=1)
        fused_emb = image_emb

        # расчёт значений целевой переменной моделью-регрессором
        result = self.regressor(fused_emb)
        
        return result

"""  
Функция валидации. 
"""
# определим код для валидации
def validate(model, val_loader, device):
    # переводим модель в режим инференса
    model.eval()
    # инициализируем пустой список абсолютных отклонений
    all_absolute_errors = []
    
    # отключаем расчёт градиентов
    with torch.no_grad():
        # проходим по батчам
        for _, batch in enumerate(
            tqdm(val_loader, total=len(val_loader), leave=False, desc="validate")
            ): 
            # входные данные для работы моделей
            inputs = {
                # 'input_ids': batch['input_ids'].to(device), 
                # 'attention_mask': batch['attention_mask'].to(device), 
                'image': batch['image'].to(device)
            }
            # переменные для расчёта MAE
            calories = batch['calories'].to(device)
            mass = batch['mass'].to(device)

            # получим результат работы модели (калорийность на грамм)
            result = model(**inputs)
            # print(f"Result_val shape: {result.shape}") 

            # рассчитаем калорийность блюда целиком
            # results_total = result * mass
            results_total = result
            # рассчитаем абсолютные отклонения
            absolute_errors = torch.abs(results_total - calories)
            # добавим рассчитанные абс. отклонения в список их значений по эпохе
            all_absolute_errors.extend(absolute_errors.tolist())
        
        # преобразуем список значений абсолютных отклонений так, чтобы не было вложений
        all_absolute_errors = list(itertools.chain.from_iterable(all_absolute_errors))
        # рассчитываем MAE для эпохи
        val_mae = sum(all_absolute_errors) / len(all_absolute_errors)

    return val_mae

"""  
Функция обучения. 

Скорость обучения задана отдельно для текстовой модели, модели для фото и 
модели регрессии, значения – как в последем задании модуля. 
Как текстовую модель по умолчанию рассматриваем bert-base-uncased, 
как модель обработки изображений – tf_efficientnet_b0 (как в последнем задании модуля). 
Если покопаться, наверняка можно будет найти что-то лучшее. 

Оптимизатор – AdamW, функция потерь – L1Loss. 

Мультимодальная модель на выходе должна давать количество калорий на грамм блюда. 
MAE рассчитываем вручную умножением массы блюда на его калорийность на грамм. 
Затем вычисляем разность между рассчитанным и реальным количеством калорий, 
разности усредняем. 
"""

# определим функцию обучения
def train(config, train_dataset, val_dataset):
    # определить, с каким устройством (cpu/cuda) работаем
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # перенесём модель на устройство
    model = MultimodalModel(config).to(DEVICE)
    
    # Оптимизатор с параметрами lr из конфига; особое внимание к float, 
    # т.к. из конфига lr почему-то грузится как str. 
    optimizer = optim.AdamW([
        # {'params': model.text_model.parameters(), 'lr': float(config['text_lr'])}, 
        {'params': model.image_model.parameters(), 'lr': float(config['image_lr'])},
        {'params': model.regressor.parameters(), 'lr': float(config['regressor_lr'])}
    ])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    # определим функцию вычисления потерь
    criterion = nn.L1Loss()
    
    # определим загрузчики для обучающего и тестового наборов
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn), 
        worker_init_fn=np.random.seed(config_notebook['seed']), 
        generator=g
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn), 
        worker_init_fn=np.random.seed(config_notebook['seed']), 
        generator=g
    )
    
    # обучение
    # определим метрику
    # mae_metric = MeanAbsoluteError().to(DEVICE)
    # цикл прохода по эпохам
    for epoch in range(config['epochs']):
        # перевод модели в режим обучения
        model.train()
        # инициируем значение суммарных потерь
        total_loss = 0.0
        # инициируем пустой список для значений абсолютных ошибок
        all_absolute_errors = []

        # цикл прохода по батчам внутри эпохи
        for _, batch in enumerate(
            tqdm(
                train_loader, 
                total=len(train_loader), 
                leave=False, 
                desc=f"train epoch: {epoch + 1}"
                )
            ): 
            # входные данные для работы моделей
            inputs = {
                # 'input_ids': batch['input_ids'].to(DEVICE), 
                # 'attention_mask': batch['attention_mask'].to(DEVICE), 
                'image': batch['image'].to(DEVICE)
            }
            # целевая переменная
            # calories_per_g = batch['calories_per_g'].to(DEVICE)
            # переменные для расчёта MAE
            calories = batch['calories'].to(DEVICE)
            mass = batch['mass'].to(DEVICE)

            # обнулим градиенты
            optimizer.zero_grad()

            # получим результат работы модели (калорийность на грамм)
            results = model(**inputs)
            # рассчитаем калорийность блюда целиком
            # results_total = results * mass
            results_total = results
            # рассчитаем абсолютные отклонения
            absolute_errors = torch.abs(results_total - calories)
            # добавим рассчитанные абс. отклонения в список их значений по эпохе
            all_absolute_errors.extend(absolute_errors.tolist())

            # рассчитаем потери
            # loss = criterion(results.squeeze(-1), calories_per_g)
            loss = criterion(results.squeeze(-1), calories)
            # выполним обратный проход
            loss.backward()
            # обновим веса
            optimizer.step()

            # добавляем потери по батчу в суммарные
            total_loss += loss.item()

        # преобразуем список значений абсолютных отклонений так, чтобы не было вложений
        all_absolute_errors = list(itertools.chain.from_iterable(all_absolute_errors))
        # рассчитываем MAE для эпохи
        train_mae = sum(all_absolute_errors) / len(all_absolute_errors)

        # рассчитываем MAE для валидационного набора
        val_mae = validate(model, val_loader, device=DEVICE)
        # выводим значения метрик (MAE и loss)
        print(f"Epoch {epoch+1}/{config['epochs']} | train MAE: {train_mae:.4f} | val MAE: {val_mae:.4f}")
        print(f"Epoch {epoch+1}/{config['epochs']} | train loss: {total_loss / len(train_loader):.4f}")
        scheduler.step(val_mae) 

        # Если целевое значение MAE достигнуто, то...
        if val_mae < VAL_MAE or epoch == config['epochs'] - 1: 
            # ...выводим на экран сообщение об этом, ...
            print('-' * 15)
            print('Обучение завершено.')
            # ...подаём звуковой сигнал, ... 
            # beeper = notes.beeps()
            # beeper.hear('C4', 200)
            # beeper.hear('C4', 200)
            # beeper.hear('C4', 200)
            # beeper.hear('G4', 600)
            # ...сохраняем веса и ... 
            torch.save(
                model.state_dict(), 
                os.path.join(os.getcwd(), config_notebook['save_path'])
                )
            # завершаем работу функции 'train'. 
            break
if __name__ == '__main__':
    train(config_notebook, train_dataset=ds_train, val_dataset=ds_val)