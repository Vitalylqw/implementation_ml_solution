# implementation_ml_solution
Курсовой проект по программе Машинное обучение в бизнесе. geekbrains.ru

11.02.2021.

В качестве данных будет использован датасет из банковской области. Задача, предсказние дефолта клиента.
Задача бианрной клакссификации

Задача

Требуется, на основании имеющихся данных о клиентах банка, построить модель, используя обучающий датасет, для прогнозирования невыполнения долговых обязательств по текущему кредиту. Разработать среду ,на основе web сервиса, для получения данных (предсказаний) в реальном времени для работы в реальных условиях.

Наименование файлов с данными

data.csv - обучающий датасет


Целевая переменная

Credit Default - факт невыполнения кредитных обязательств

Метрика качества

F1-score (sklearn.metrics.f1_score)

Требования к решению

Целевая метрика

F1 > 0.5. precision и recall >0.5
Метрика оценивается по качеству прогноза для главного класса (1 - просрочка по кредиту)


Обзор данных
Описание датасета

Home Ownership - домовладение  
Annual Income - годовой доход  
Years in current job - количество лет на текущем месте работы  
Tax Liens - налоговые обременения  
Number of Open Accounts - количество открытых счетов  
Years of Credit History - количество лет кредитной истории  
Maximum Open Credit - наибольший открытый кредит  
Number of Credit Problems - количество проблем с кредитом  
Months since last delinquent - количество месяцев с последней просрочки платежа  
Bankruptcies - банкротства  
Purpose - цель кредита  
Term - срок кредита  
Current Loan Amount - текущая сумма кредита  
Current Credit Balance - текущий кредитный баланс  
Monthly Debt - ежемесячный долг  
Credit Default - факт невыполнения кредитных обязательств (0 - погашен вовремя, 1 - просрочка)  

Обоснование модели в файле model_development.ipynb

Работа с программой:  
api сервис запускается на локальном web сервере и доступен по адресу http://127.0.0.1:5000/ , на локальной машине  
Что бы начать работать, необходимо:
1. разместить в папке дата, в корне проекта файл train.csv, для обучения моделей, файл включает таргет  
2. При необходимости внести изменения в настройки (файл ather_data.py). На данный момент там оптимальныет настройки для текущих данных
В случае каких то изменений, там можно поменять как структуру данных, так и параметры и гиперпараметры модели. Обоснование применения текущих параметров находиться в файле model_development.ipynb.
3. Далее нужно запустить обучение модели. Для этого нужно запсутить на исполнение файл fit_model.py. 
После этого в папке data в корне проекта появятся необходимые для работы файлы
4. Что бы запустить сервер необходимо   запустить файл server.py
6. Что бы получить данные необходимо отправить get или post запрос по адресу url = 'http://127.0.0.1:5000/predict_one/'
   и передать в запросе в качестве параметров одну строку с данными, (без таргета).  
   7. в ответ придет сообщение с указанием статуса и результата. Если статус отрицательный, будет указана ошибка
    
Для тестированя клиента есть файл sender.py
   
  

	
