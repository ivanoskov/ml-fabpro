import users_ratings, movies_metadata from dataset.py

import numpy as np
import pandas as pd

"""
Простая рекомендательная система на основе контента (content-based filtering)

Алгоритм работы:
1. Система использует сходство жанров для определения похожести фильмов
2. Для каждого непросмотренного фильма:
   - Находим схожесть с каждым просмотренным фильмом пользователя
   - Учитываем оценки пользователя, умножая на коэффициент схожести
   - Вычисляем среднюю оценку
3. Рекомендуем фильмы с наивысшей расчетной оценкой

Метрики:
- Схожесть фильмов: коэффициент Жаккара (пересечение/объединение жанров)
- Итоговая оценка: средневзвешенная оценка по схожести фильмов
"""

class SimpleRecommender:
    def __init__(self, users_data, movies_data):
        """
        Инициализация рекомендательной системы
        :param users_data: словарь с оценками пользователей
        :param movies_data: словарь с информацией о фильмах
        """
        self.users_data = users_data  # Сохраняем данные пользователей
        self.movies_data = movies_data  # Сохраняем данные о фильмах
        
    def get_movie_similarity(self, movie1, movie2):
        """
        Вычисляем схожесть двух фильмов по их жанрам используя коэффициент Жаккара
        Коэффициент Жаккара = (количество общих жанров) / (количество всех уникальных жанров)
        Например: 
        movie1 = ['фантастика', 'боевик']
        movie2 = ['фантастика', 'триллер']
        Общий жанр: ['фантастика']
        Все жанры: ['фантастика', 'боевик', 'триллер']
        Схожесть = 1/3 ≈ 0.33
        """
        # Преобразуем списки жанров в множества для удобства операций
        genres1 = set(self.movies_data[movie1]['жанр'])
        genres2 = set(self.movies_data[movie2]['жанр'])
        
        # Вычисляем коэффициент Жаккара
        # intersection() - находит общие элементы множеств
        # union() - объединяет множества без дубликатов
        return len(genres1.intersection(genres2)) / len(genres1.union(genres2))
    
    def recommend_for_user(self, user, n_recommendations=2):
        """
        Формируем рекомендации для пользователя
        :param user: имя пользователя
        :param n_recommendations: количество рекомендаций
        :return: список кортежей (название_фильма, оценка)
        """
        # Находим фильмы, которые пользователь еще не смотрел
        # set() преобразует списки в множества для быстрого поиска разницы
        unwatched_movies = set(self.movies_data.keys()) - set(self.users_data[user].keys())
        
        # Словарь для хранения рассчитанных оценок
        recommendations = {}
        
        # Для каждого непросмотренного фильма
        for movie in unwatched_movies:
            score = 0  # Накопитель итоговой оценки
            watched_count = 0  # Счетчик похожих фильмов
            
            # Сравниваем с каждым просмотренным фильмом
            for watched_movie, rating in self.users_data[user].items():
                # Получаем коэффициент схожести
                similarity = self.get_movie_similarity(movie, watched_movie)
                
                # Если фильмы имеют что-то общее
                if similarity > 0:
                    # Умножаем схожесть на оценку пользователя
                    # Чем больше схожесть и выше оценка, тем больше вклад
                    score += similarity * rating
                    watched_count += 1
            
            # Если нашлись похожие фильмы
            if watched_count > 0:
                # Вычисляем среднюю оценку
                recommendations[movie] = score / watched_count
                
        # Сортируем фил��мы по убыванию оценки
        # key=lambda x: x[1] - сортировка по второму элементу кортежа (оценке)
        # reverse=True - сортировка по убыванию
        # [:n_recommendations] - берем только запрошенное количество рекомендаций
        return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

# Создаем объект рекомендательной системы
recommender = SimpleRecommender(users_ratings, movies_metadata)
user = 'Алексей'

# Выводим рекомендации
print(f"Рекомендации для {user}:")
# Получаем и выводим топ рекомендаций с оценками и жанрами
for movie, score in recommender.recommend_for_user(user):
    print(f"{movie}: {score:.2f} - {movies_metadata[movie]['жанр']}")

# В конце добавим более информативный вывод:
for user in users_ratings.keys():
    print(f"\nАнализ предпочтений пользователя {user}:")
    
    # Подсчет любимых жанров
    genre_ratings = {}
    for movie, rating in users_ratings[user].items():
        for genre in movies_metadata[movie]['жанр']:
            if genre not in genre_ratings:
                genre_ratings[genre] = []
            genre_ratings[genre].append(rating)
    
    print("Любимые жанры:")
    for genre, ratings in genre_ratings.items():
        avg_rating = sum(ratings) / len(ratings)
        if avg_rating >= 4:  # Выводим только жанры с высокой оценкой
            print(f"- {genre}: {avg_rating:.2f}")
    
    print("\nРекомендации:")
    for movie, score in recommender.recommend_for_user(user, n_recommendations=3):
        print(f"- {movie} (схожесть: {score:.2f})")
        print(f"  Жанры: {', '.join(movies_metadata[movie]['жанр'])}")
    print("-" * 50)
