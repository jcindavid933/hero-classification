from sklearn import tree

# Classifier used here: A simple decision tree
clf = tree.DecisionTreeClassifier()

# Features of the person: [Gender: male(1) or female(0), mask: yes(1) or no(0), cape: yes(1) or no(0), human: yes(1) or no(0)] --> HERO VS VILLIAN
X = [[1, 1, 1, 1], # Batman
    [1, 0, 0, 1], # Alfred
    [1, 1, 0, 1], # Iron Man
    [0, 1, 0, 1], # Catwoman
    [1, 0, 0, 1], # Joker
    [1, 0, 1, 0], # Thor
    [0, 0, 0, 1], # Black Widow
    [1, 0, 0, 0], # Thanos
    [1, 1, 1, 1], # Magneto
    [1, 0, 1, 0], # DORMAMMU
    [0, 0, 0, 0], # Gamora
    [1, 0, 0, 0], # Mantis
    [1, 0, 0, 0], # Groot
    [1, 0, 0, 0], # Drax the Destroyer
    [1, 0, 0, 1], # ERIK “KILLMONGER” STEVENS
    [0, 0, 0, 0], # NEBULA
    [1, 0, 1, 0], # RONAN
    [0, 0, 1, 0], # HELA
    [1, 0, 0, 0], # SURTUR
    [0, 0, 1, 0], # Maleficent
    [1, 1, 0, 1], # Captain America
    [0, 0, 1, 1]] # Captain Marvel


Y = ['hero', 'hero', 'hero', 'villain', 'villain', 'hero', 'hero', 'villain',
     'villain', 'villain', 'hero', 'hero', 'hero', 'hero', 'villain', 'villain', 'villain', 'villain', 'villain', 'villain', 'hero', 'hero']


# Training the data with the features and labels
clf = clf.fit(X, Y)

prediction = clf.predict([[0,1,0,0]])

# Predicting the person based on the provided data

print(prediction)
