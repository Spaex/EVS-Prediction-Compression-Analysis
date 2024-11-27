"""
Dieses Skript führt eine Datenkompressionsanalyse mit Huffman-Codierung auf ereignisbasierten Sensordaten durch.
Es liest Daten aus einer HDF5-Datei, fügt den Daten Rauschen hinzu, berechnet Kompressionsraten und stellt die Ergebnisse grafisch dar.
Funktionen:
    huffman_encode(data):
        Codiert die Eingabedaten mit Huffman-Codierung.
Hauptskript:
    - Liest Ereignisdaten aus einer HDF5-Datei.
    - Teilt die Daten in Zeit, x, y auf.
    - Fügt den Daten basierend auf angegebenen Genauigkeiten Rauschen hinzu.
    - Berechnet den Root Mean Square Error (RMSE) für x, y und Zeit.
    - Codiert die Delta-Werte von x, y und Zeit mit Huffman-Codierung.
    - Berechnet Kompressionsraten für verschiedene Genauigkeiten.
    - Stellt die Kompressionsraten gegen die Genauigkeiten grafisch dar.
Variablen:
    filename (str): Pfad zur HDF5-Datei mit den Ereignisdaten.
    key (str): Schlüssel zum Zugriff auf die Ereignisdaten in der HDF5-Datei.
    data (ndarray): Array, das die Ereignisdaten enthält.
    time (ndarray): Array, das die Zeitwerte der Ereignisse enthält.
    x (ndarray): Array, das die x-Koordinaten der Ereignisse enthält.
    y (ndarray): Array, das die y-Koordinaten der Ereignisse enthält.
    time_difference (int): Differenz zwischen dem letzten und dem ersten Zeitwert.
    max (ndarray): Maximalwerte für x, y und Zeit.
    bits (ndarray): Bitlängen für x, y und Zeit.
    x_bitsize (int): Größe von x in Bits.
    y_bitsize (int): Größe von y in Bits.
    time_bitsize (int): Größe von Zeit in Bits.
    mu (int): Mittelwert für die Gauß-Verteilung.
    accuracies (ndarray): Array von Genauigkeitswerten.
    accuracy_real_x (list): Liste der realen Genauigkeiten für x.
    accuracy_real_y (list): Liste der realen Genauigkeiten für y.
    accuracy_real_time (list): Liste der realen Genauigkeiten für Zeit.
    x_compression_rates (list): Liste der Kompressionsraten für x.
    y_compression_rates (list): Liste der Kompressionsraten für y.
    time_compression_rates (list): Liste der Kompressionsraten für Zeit.
    rmses_x (list): Liste der RMSE-Werte für x.
    rmses_y (list): Liste der RMSE-Werte für y.
    rmses_time (list): Liste der RMSE-Werte für Zeit.
    delta_x (ndarray): Delta-Werte für x.
    delta_y (ndarray): Delta-Werte für y.
    delta_time (ndarray): Delta-Werte für Zeit.
    delta_x_encoded (str): Huffman-codierte Delta-Werte für x.
    delta_y_encoded (str): Huffman-codierte Delta-Werte für y.
    delta_time_encoded (str): Huffman-codierte Delta-Werte für Zeit.
    accuracies_x (ndarray): Array der Genauigkeiten für x.
    accuracies_y (ndarray): Array der Genauigkeiten für y.
    accuracies_time (ndarray): Array der Genauigkeiten für Zeit.
    accuracies_ges (ndarray): Array der kombinierten Genauigkeiten.
    compression_rates (list): Liste der kombinierten Kompressionsraten.
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from collections import Counter
import platform

# Implementierung des Huffman Codes
def huffman_encode(data):
    # Zähle die Häufigkeit jedes Symbols
    freq = Counter(data)
    
    # Erstelle eine Prioritätswarteschlange für die Symbole
    pq = []
    for symbol, count in freq.items():
        pq.append((count, symbol))
    pq.sort()
    
    # Baue den Huffman-Baum
    while len(pq) > 1:
        freq1, symbol1 = pq.pop(0)
        freq2, symbol2 = pq.pop(0)
        merged_freq = freq1 + freq2
        merged_symbol = (symbol1, symbol2)
        pq.append((merged_freq, merged_symbol))
        pq = sorted(pq, key=lambda x: x[0])
    
    # Erstelle das Huffman-Codierungswörterbuch
    codes = {}
    def build_codes(node, code):
        if isinstance(node, tuple):
            build_codes(node[0], code + '0')
            build_codes(node[1], code + '1')
        else:
            codes[node] = code
    build_codes(pq[0][1], '')
    
    # Codiere die Daten
    encoded_data = ''.join(codes[symbol] for symbol in data)

    # Print the Huffman dictionary
    #print("Huffman Dictionary:")
    #for symbol, code in codes.items():
    #    print(f"{symbol}: {code}")         # Optional kann hier das Huffman-Wörterbuch ausgegeben werden

    
    return encoded_data


if platform.system() == "Windows":
    filename = "D:\\Drive\\Bachelorarbeit\\Jupyter Lab\\rec1487857941.hdf5.exported.hdf5"
else:
    filename = "rec1487857941.hdf5.exported.hdf5" # Bei Bedarf können hier abhängig von Win/Linux die Dateipfade angepasst werden


# Einlesen der Daten vom Key 'event' / DDD20
dataset = h5py.File(filename, 'r')
key = 'event'

data = np.array(dataset[key])

# Spalten aufteilen: time, x, y, pol
time = data[:100000, 0]
x = data[:100000, 1]
y = data[:100000, 2]
time_difference = time[-1] - time[0]

time = time - time[0] # Starte Zeit bei 0
max = np.array([1279, 719, 63]) # Maximale ∆E für eine Sensorgröße von 1280x720 Pixeln im EVT2.0 Format
bits = np.array([11, 11, 6]) # Bitlänge der einzelnen Elemente X,Y und Zeit

x_bitsize = x.size * bits[0]    # Größe von X in Bits
y_bitsize = y.size * bits[1]    # Größe von Y in Bits
time_bitsize = time.size * bits[2] + 28 * (1 + int(time[-1] / 64)) # Größe von Zeit in Bits

# Parameter für die Gauß-Verteilung
mu = 0

accuracies = np.arange(0, 1.01, 0.01)
#accuracies = np.concatenate((np.arange(0, 0.95, 0.01), np.arange(0.95, 1.001, 0.001))) # Optional kann hier die Genauigkeit feiner festgelegt werden für die letzten 5% Genauigkeit (schönerer Plot)
accuracy_real_x = []
accuracy_real_y = []
accuracy_real_time = []

x_compression_rates = []
y_compression_rates = []
time_compression_rates = []
rmses_x = []
rmses_y = []
rmses_time = []

for accuracy in accuracies:

    if accuracy == 1:
        x_noise = np.zeros(x.size)
        y_noise = np.zeros(y.size)
        time_noise = np.zeros(time.size)
    else:
        x_noise = np.round(np.clip(np.round(np.random.normal(mu, max[0]/4, x.size)), -max[0], max[0])) # Gauß-Verteilung
        #x_noise = np.random.randint(-max[0], (max[0] + 1), x.size) # Gleichverteilung
        if accuracy == 0:
            plt.hist(x_noise, bins=max[0]*2+1, edgecolor='black')
            plt.title('Histogramm von x_noise')
            plt.xlabel('Wert')
            plt.ylabel('Auftrittshäufigkeit')
            plt.savefig('x_noise_distribution.png')
        num_zeros_x = int(accuracy * len(x_noise))
        indices_x = np.random.choice(len(x), num_zeros_x, replace=False)
        x_noise[indices_x] = 0
        num_zeros_x = np.sum(x_noise == 0)
        unique_numbers_x = len(np.unique(x_noise))
        y_noise = np.clip(np.round(np.random.normal(mu, max[1]/4, y.size)), -max[1], max[1])    # Gauß-Verteilung
        #y_noise = np.random.randint(-max[1], max[1] + 1, y.size)   # Gleichverteilung
        if accuracy == 0:
            plt.figure()
            plt.hist(y_noise, bins=max[1]*2 +1, edgecolor='black')
            plt.title('Histogramm von y_noise')
            plt.xlabel('Wert')
            plt.ylabel('Auftrittshäufigkeit')
            plt.savefig('y_noise_distribution.png')
        num_zeros_y = int(accuracy * len(y_noise))
        indices_y = np.random.choice(len(y), num_zeros_y, replace=False)
        y_noise[indices_y] = 0
        #time_noise = np.random.randint(-max[2], max[2], time.size) # Gleichverteilung
        time_noise = np.clip(np.round(np.random.normal(mu, max[2]/4, time.size)), -max[2], max[2]) # Gauß-Verteilung
        if accuracy == 0:
            plt.figure()
            plt.hist(time_noise, bins=126, edgecolor='black')
            plt.title('Histogramm von time_noise')
            plt.xlabel('Wert')
            plt.ylabel('Auftrittshäufigkeit')
            plt.savefig('time_noise_distribution.png')
        num_zeros_time = int(accuracy * len(time_noise))
        indices_time = np.random.choice(len(time), num_zeros_time, replace=False)
        time_noise[indices_time] = 0

    delta_x = x - x
    delta_y = y - y
    delta_time = time - time
    delta_x = delta_x + x_noise
    delta_y = delta_y + y_noise
    delta_time = delta_time + time_noise
    delta_x_encoded = huffman_encode(delta_x)
    delta_y_encoded = huffman_encode(delta_y)
    delta_time_encoded = huffman_encode(delta_time)
    rmses_x.append(np.sqrt(np.mean(delta_x**2)))
    rmses_y.append(np.sqrt(np.mean(delta_y**2)))
    rmses_time.append(np.sqrt(np.mean(delta_time**2)))
    num_zeros_delta_x = np.sum(delta_x == 0)
    num_zeros_delta_y = np.sum(delta_y == 0)
    num_zeros_delta_time = np.sum(delta_time == 0)
    accuracy_real_x.append(num_zeros_delta_x/len(delta_x))
    accuracy_real_y.append(num_zeros_delta_y/len(delta_y))
    accuracy_real_time.append(num_zeros_delta_time/len(delta_time))
    if accuracy == 1:
        x_savings = len(x) / x_bitsize
        y_savings = len(y) / y_bitsize
        time_savings = len(time) / time_bitsize
    else:
        x_savings = len(delta_x_encoded) / x_bitsize
        y_savings = len(delta_y_encoded) / y_bitsize
        time_savings = len(delta_time_encoded) / time_bitsize
    print(f"{accuracy * 100:.2f}% bereits berechnet.")
    x_compression_rate = 1 / x_savings
    y_compression_rate = 1 / y_savings
    time_compression_rate = 1 / time_savings

    x_compression_rates.append(x_compression_rate)
    y_compression_rates.append(y_compression_rate)
    time_compression_rates.append(time_compression_rate)

accuracies_x = ((1 - (rmses_x/max[0])) + accuracy_real_x) / 2 * 100
accuracies_y = ((1 - (rmses_y/max[1])) + accuracy_real_y) / 2 * 100
accuracies_time = ((1 - (rmses_time/max[2])) + accuracy_real_time) / 2 * 100
accuracies_ges = (accuracies_x + accuracies_y + accuracies_time) / 3
compression_rates = [(x + y + t) / 3 for x, y, t in zip(x_compression_rates, y_compression_rates, time_compression_rates)]
accuracies = accuracies * 100

plt.figure(figsize=(12, 6))
plt.plot(accuracies_x, x_compression_rates, label='X Kompressionsrate')
plt.plot(accuracies_y, y_compression_rates, label='Y Kompressionsrate')
plt.plot(accuracies_time, time_compression_rates, label='Zeit t Kompressionsrate')
plt.plot(accuracies_ges, compression_rates, label='Gesamtkompressionsrate', linewidth=2.5) # Kompressionsrate kombiniert x und y und time
plt.xlabel('Genauigkeit [%]')
plt.ylabel('Kompressionsrate [x:1]')
plt.title('Kompressionsrate über die Prädiktorgenauigkeit')
plt.xlim(left=0)
plt.yticks(range(1, 12, 1))

plt.legend()
if platform.system() == "Windows":
    plt.savefig('compression_rate_vs_accuracy.png')
    plt.savefig('compression_rate_vs_accuracy_transparent.png', transparent=True) # Speichern des Plots als PNG-Datei zusätzlich ohne Hintergrund
else:
    plt.savefig('compression_rate_vs_accuracy.png')
    plt.savefig('compression_rate_vs_accuracy_transparent.png', transparent=True) # Zusätzlich kann hier ein weiterer Pfad angegeben werden für Linux
plt.show()
