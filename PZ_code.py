#Импортируем требуемые библиотеки
import streamlit as st
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier

#задаем название на первой странице приложения. После реализации этой части кода запустите приложение и попробуйте поменять название на главной странице и затем запустить приложение через терминал. Посмотрите как реагирует приложение на изменение названия в коде.
st.write("""
# fSimple Iris Flower Prediction App.
This app predicts the **Example** type!
""")

#зададим левую колонку в приложении. Также постарайтесь поменять ее название. Запустить приложение и посмотреть как меняется это название. 
st.sidebar.header('User Input Parameters')

#зададим функцию. Определим значения ключевых параметров ('Sepal length', 'Sepal width' и другие) и датафрейм. Поменяйте их значения и посмотрите как будет реагировать приложение. 
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

#зададим переменную, содержащую входные значения
df = user_input_features()

#пропишем название параметров на лицевой страничке приложения и выведем значения этих параметров
st.subheader('User Input parameters')
st.write(df)

#зададим переменные
iris = datasets.load_iris()
X = iris.data
Y = iris.target

#классифицируем данные методом “случайного леса”
clf = RandomForestClassifier()
clf.fit(X, Y)

#спрогнозируем значения
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

#пропишем название таблицы и зададим значения выводимых параметров в приложении
st.subheader('Class labels and their corresponding index number')
st.write(iris.target_names)
#зададим предсказываемые значения
st.subheader('Prediction')
st.write(iris.target_names[prediction])
#зададим вероятность прогнозирования. То есть вероятность попадания цветка ириса в тот или иной класс. 
st.subheader('Prediction Probability')
st.write(prediction_proba)
