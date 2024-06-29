import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix

# ჩატვირთეთ მონაცემები
url = 'https://raw.githubusercontent.com/path/to/your-data.csv'
data = pd.read_csv(url)

# ფუნქცია მონაცემების დათვალიერებისთვის
def preview_data(data):
    print(data.head())
    print(data.info())
    print(data.describe())

preview_data(data)

# Task 1: მარტივი ხაზობრივი რეგრესია
# ამოცანა: გამოიყენეთ ერთი ცვლადის რეგრესიული მოდელი
X = data[['your_feature']]  # შეცვალეთ თქვენი ცვლადით
y = data['your_target']  # შეცვალეთ თქვენი სამიზნე სვეტით

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# პროგნოზირება ახალი მონაცემებით
y_pred = model.predict(X_test)

# მოდელის ეფექტიანობის გამოთვლა
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Task 2: მრავალცვლადიანი რეგრესია
# ამოცანა: გამოიყენეთ მრავალცვლადიანი რეგრესიული მოდელი
X = data[['your_feature1', 'your_feature2']]  # შეცვალეთ თქვენი ცვლადებით
y = data['your_target']  # შეცვალეთ თქვენი სამიზნე სვეტით

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# პროგნოზირება ახალი მონაცემებით
y_pred = model.predict(X_test)

# მოდელის ეფექტიანობის გამოთვლა
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Task 3: გადაწყვეტილების ხის რეგრესია
# ამოცანა: გამოიყენეთ გადაწყვეტილების ხის რეგრესიული მოდელი
X = data[['your_feature1', 'your_feature2']]  # შეცვალეთ თქვენი ცვლადებით
y = data['your_target']  # შეცვალეთ თქვენი სამიზნე სვეტით

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

# პროგნოზირება ახალი მონაცემებით
y_pred = model.predict(X_test)

# მოდელის ეფექტიანობის გამოთვლა
mse = mean_squared_error(y_test, y_pred)
print(f'MSE: {mse}')

# Task 4: ლოგისტიკური რეგრესია
# ამოცანა: გამოიყენეთ ლოგისტიკური რეგრესიის მოდელი
X = data[['your_feature1', 'your_feature2']]  # შეცვალეთ თქვენი ცვლადებით
y = data['your_target_binary']  # შეცვალეთ თქვენი ბინარული სამიზნე სვეტით

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# პროგნოზირება ახალი მონაცემებით
y_pred = model.predict(X_test)

# მოდელის ეფექტიანობის გამოთვლა
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Task 5: გადაწყვეტილების ხის კლასიფიკაცია
# ამოცანა: გამოიყენეთ გადაწყვეტილების ხის კლასიფიკაციის მოდელი
X = data[['your_feature1', 'your_feature2']]  # შეცვალეთ თქვენი ცვლადებით
y = data['your_target_binary']  # შეცვალეთ თქვენი ბინარული სამიზნე სვეტით

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# პროგნოზირება ახალი მონაცემებით
y_pred = model.predict(X_test)

# მოდელის ეფექტიანობის გამოთვლა
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)

