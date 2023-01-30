import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB


Categories = ['Age', 'Game', 'Genre', 'Platform']
GameData = pd.read_csv('GameAnalysis.csv', header=None, names=Categories)
print(GameData)
print(GameData.shape)

print("***********************************************")
print("***********************************************")

inputs = GameData.drop('Platform', axis='columns')
target = GameData['Platform']

print(target)

print("***********************************************")
print("***********************************************")


Age = LabelEncoder()
Game = LabelEncoder()
Genre = LabelEncoder()

inputs['Age_new'] = Age.fit_transform(inputs['Age'])
inputs['Game_new'] = Age.fit_transform(inputs['Game'])
inputs['Genre_new'] = Age.fit_transform(inputs['Genre'])

print("***********************************************")
print("***********************************************")

inputs_new = inputs.drop(['Age', 'Game', 'Genre'], axis='columns')
print("New Encoder List")
print(inputs_new)

print("***********************************************")
print("***********************************************")

model = GaussianNB()
print(model.fit(inputs_new, target))
print(model.score(inputs_new, target))

print("***********************************************")
print("***********************************************")

expectations = inputs_new
predictions = model.predict([[1,2,1], [2,0,3]])
# According to EncoderList [1 = age between 12 and 18, 2 = Fifa, 1 = FPS], [2 = age between 25 and 30, 0 = Age of Empire, 3 = Sport]
print(predictions)



