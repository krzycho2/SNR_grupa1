import numpy as np

# Perceptron - będzie rozwiązywał zadanie klasyfikacji binarnej
# Dla przykładu: Przyporządkowanie punktów do zbioru - klasy 1 i 2
# Zbiór trenujący: 
# X - współrzędne punktów, 
# Y : {-1,1} - wskazuje czy punkt należy do zbioru czy nie
# 
def create_perceptron(X, Y):
    """
    Metoda trenująca. Zwraca wektor wag 
    """
    w = np.zeros(len(X[0]))          # Wektor wag o długości X
    print("Wagi: " + str(w) + "\n")
    eta = 1                             # Współczynnik uczenia
    epoki = 20                          # Liczba epok - okresów uczenia

    for epoka in range(epoki):
        print("Epoka " + str(epoka) + "\n")
        for i,x in enumerate(X):
            if(np.matmul(X[i], w) * Y[i]) <= 0: # Y[i] = funkcja aktywacji 
                w = w + eta*X[i]*Y[i]
                print("Nowe wagi: " + str(w) + "\n")
            else:
                print("Wagi bez zmian\n")
    return w

def test_perceptron(X, wagi):
    Y = 0
    for i, x in enumerate (X):
        if(np.matmul(X[i], wagi)) <= 0: # Funkcja aktywacji
             Y = -1
        else:
            Y = 1
        print(str(X[i]) + " zwraca " + str(Y) + "\n")

# Uczenie
# Wektory uczące X_train i Y_train
X_train = np.array([
    [1, -1],
    [2, -1],
    [10, 2],
    [-7, 1],
    [0, -10]
])
Y_train = np.array([-1 ,-1, 1, 1, -1])
wagi = create_perceptron(X_train, Y_train)

#Testowanie
print("Test perceptronu")
x_test = np.array([
    [0,1],      # Zwraca 1
    [1,0],      # Zwraca -1
    [2,-2]      # Zwraca -1
    ])
test_perceptron(x_test, wagi)
