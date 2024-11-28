"""
Dieses Programm stellt die benötigte Bandbreite für die Übertragung von Ereignisdaten dar und vergleicht sie mit verschiedenen Komprimierungsmethoden.
Module:
    - h5py: Zum Lesen und Schreiben von HDF5-Dateien.
    - matplotlib.pyplot: Zum Erstellen von Plots und Animationen.
    - numpy: Für numerische Operationen.
    - scipy.io: Zum Speichern von MATLAB-Dateien.
    - collections.Counter: Zum Zählen der Häufigkeit von Symbolen.
    - platform: Zum Erkennen des Betriebssystems.
    - matplotlib.animation.FuncAnimation: Zum Erstellen von Animationen.
    - heapq: Für die Prioritätswarteschlange in der Huffman-Codierung.
    - graphviz: Zum Visualisieren des Huffman-Baums.
    - pandas: Zum Erstellen und Verarbeiten von DataFrames.
Funktionen:
    - huffman_encode(data): Führt die Huffman-Codierung auf den gegebenen Daten durch.
    - binary_to_bit_length(binary_array): Konvertiert ein Array von Binärzahlen in deren Bitlängen.
    - binary_array_to_decimal(binary_array): Konvertiert ein Array von Binärzahlen in Dezimalzahlen.
    - visualize_huffman_tree(codes): Visualisiert den Huffman-Baum basierend auf den gegebenen Codes.
    - huffman_decode(encoded_data, codes): Dekodiert die Huffman-codierten Daten basierend auf den gegebenen Codes.
Hauptprogramm:
    - Liest Ereignisdaten aus einer HDF5-Datei basierend auf dem Betriebssystem.
    - Führt eine Huffman-Codierung auf einem Testdatensatz durch und erstellt eine Tabelle der Symbolhäufigkeiten.
    - Visualisiert den Huffman-Baum und speichert ihn als Bild.
    - Kodiert und dekodiert die Zeit-, x- und y-Daten der Ereignisse.
    - Berechnet die Bitrate für verschiedene Komprimierungsmethoden und erstellt eine Animation der Bitraten über die Zeit.
"""
import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from collections import Counter
import platform
from matplotlib.animation import FuncAnimation
import heapq
from collections import Counter
from graphviz import Digraph
import pandas as pd

def huffman_encode(data):
    # Zähle die Häufigkeit jedes Symbols
    freq = Counter(data)
    
    # Erstelle eine Prioritätswarteschlange für die Symbole
    pq = [[count, [symbol, ""]] for symbol, count in freq.items()]
    heapq.heapify(pq)
    
    # Baue den Huffman-Baum
    while len(pq) > 1:
        lo = heapq.heappop(pq)
        hi = heapq.heappop(pq)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(pq, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    
    # Erstelle das Huffman-Codierungswörterbuch
    codes = {}
    for pair in heapq.heappop(pq)[1:]:
        codes[pair[0]] = pair[1]
    
    # Codiere die Daten als Array
    encoded_data_array = [codes[symbol] for symbol in data]
    
    return encoded_data_array

def binary_to_bit_length(binary_array):
    return [len(b) for b in binary_array]

def binary_array_to_decimal(binary_array):
    return [int(b, 2) for b in binary_array]

if platform.system() == "Windows":
    with h5py.File("driving_sample.hdf5", "r") as f:
        evts = f["CD"]["events"]
        time = evts['t'][:]
        x = evts['x'][:]
        y = evts['y'][:]
else:
    with h5py.File("../Python/driving_sample.hdf5", "r") as f:
        evts = f["CD"]["events"]
        time = evts['t'][:]
        x = evts['x'][:]
        y = evts['y'][:]

#Testdatensatz für die Huffman-Codierung
test = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # 19 mal 0
 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,  # 23 mal 1
 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  # 15 mal 2
 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  # 14 mal 3
 4, 4, 4, 4, 4, 4, 4, 4,  # 8 mal 4
 5, 5, 5, 5, 5, 5, 5,  # 7 mal 5
 6, 6, 6, 6, 6,  # 5 mal 6
 7, 7, 7, 7, 7,  # 5 mal 7
 8, 8, 8,  # 3 mal 8
 9, 9, 9, 9]  # 4 mal 9
# Berechne die Häufigkeit der Symbole im Testdatensatz
freq = Counter(test)

# Erstelle eine DataFrame für die Tabelle
df = pd.DataFrame(list(freq.items()), columns=['ΔE', 'Absolute Anzahl'])
df['Prozentsatz'] = (df['Absolute Anzahl'] / len(test)) * 100

# Sortiere die Tabelle nach Symbolen
df = df.sort_values(by='ΔE').reset_index(drop=True)

# Runden der Werte
df['Absolute Anzahl'] = df['Absolute Anzahl'].astype(int)  # Ganzzahlen sicherstellen
df['Prozentsatz'] = df['Prozentsatz'].round(2)  # Prozentsatz auf 2 Nachkommastellen runden

# Erstelle die Tabelle als Bild
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Speichere die Tabelle als Bild
plt.savefig('Tabelle_Verteilung_von_ΔE.png', dpi=300, bbox_inches='tight')

encoded_test = huffman_encode(test)
print("Original:", test)
print("Encoded:", encoded_test)
# Ausgabe der Huffman-Tabelle mit den jeweiligen Codes für die jeweiligen Elemente
huffman_table = {symbol: code for symbol, code in zip(test, encoded_test)}
print("Huffman-Tabelle:")
for symbol, code in huffman_table.items():
    print(f"Symbol: {symbol}, Code: {code}")
encoded_test_decimal = binary_to_bit_length(encoded_test)
def visualize_huffman_tree(codes):
    dot = Digraph()
    dot.attr('node', shape='circle')
    
    # Erstelle Knoten für jedes Symbol und seine Huffman-Codierung
    for symbol, code in codes.items():
        current_node = ''
        for bit in code:
            next_node = current_node + bit
            if not dot.node(next_node):
                dot.node(next_node)
                if current_node:
                    dot.edge(current_node, next_node, label=bit)
            current_node = next_node
        dot.node(current_node, label=str(symbol), shape='plaintext')
    
    return dot

# Visualisiere den Huffman-Baum
codes = {symbol: code for symbol, code in zip(test, encoded_test)}
huffman_tree = visualize_huffman_tree(codes)
huffman_tree.render('huffman_tree', format='png', cleanup=True)
# Erstelle eine DataFrame für das Huffman-Wörterbuch
huffman_df = pd.DataFrame(list(codes.items()), columns=['Symbol', 'Code'])

# Erstelle die Tabelle als Bild
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=huffman_df.values, colLabels=huffman_df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# Speichere die Tabelle als Bild ohne weißen Rand oben und unten
plt.savefig('Huffman-Dictionary.png', dpi=300, bbox_inches='tight')
print("Encoded (Bitlänge):", encoded_test_decimal)


# Spalten aufteilen: time, x, y, pol
time = time - time[0]
time_noise = np.diff(time).astype(np.int32)
x_noise = np.diff(x).astype(np.int32)
y_noise = np.diff(y).astype(np.int32)
print("Maximum time noise:", np.max(time_noise))

time_encoded = huffman_encode(time_noise)
time_encoded_decimal = binary_to_bit_length(time_encoded)

print("Finished encoding time")
x_encoded = huffman_encode(x_noise)
x_encoded_decimal = binary_to_bit_length(x_encoded)
print("Finished encoding x")
y_encoded = huffman_encode(y_noise)
y_encoded_decimal = binary_to_bit_length(y_encoded)
print("Finished encoding y")


print("Maximum time encoded:", np.max(time_encoded_decimal))
print("Maximum x encoded:", np.max(x_encoded_decimal))
print("Maximum y encoded:", np.max(y_encoded_decimal))

def huffman_decode(encoded_data, codes):
    reverse_codes = {v: k for k, v in codes.items()}
    decoded_data = []
    current_code = ""
    for bit in "".join(encoded_data):
        current_code += bit
        if current_code in reverse_codes:
            decoded_data.append(reverse_codes[current_code])
            current_code = ""
    return decoded_data

# Erstelle das Huffman-Codierungswörterbuch für die Dekodierung
codes_time = {symbol: code for symbol, code in zip(time_noise, time_encoded)}
codes_x = {symbol: code for symbol, code in zip(x_noise, x_encoded)}
codes_y = {symbol: code for symbol, code in zip(y_noise, y_encoded)}

# Dekodiere die Daten
time_decoded = huffman_decode(time_encoded, codes_time)
print("Time decoded")
x_decoded = huffman_decode(x_encoded, codes_x)
print("X decoded")
y_decoded = huffman_decode(y_encoded, codes_y)
print("Y decoded")

# Validierung
assert np.array_equal(time_noise, time_decoded), "Time decoding failed"
assert np.array_equal(x_noise, x_decoded), "X decoding failed"
assert np.array_equal(y_noise, y_decoded), "Y decoding failed"

print("Time, X, and Y decoded successfully and match the original data.")

ecf_filter_code = 0x8ECF

delta_t = 10000  # 10 ms

S = 32  # EVT 2.0
S_raw = 18 + 1 + 34  # Raw (X/Y + Pol + Time)
S_compressed = 4

N = np.zeros(int(time[-1] / delta_t))

i = int(time[-1] / delta_t)
np.round(i)
print(i)
# Zählung der Ereignisse in den Zeitschritten mit np.histogram
N, _ = np.histogram(time, bins=np.arange(0, (i + 1) * delta_t, delta_t))
print("S_SEP Calculcation started")
# Initialisiere die Summen für S_SEP
S_SEP = np.zeros(int(np.ceil(time[-1] / delta_t)))

# Variablen zur Verwaltung der Intervalle
current_interval_start = 0
current_interval_end = delta_t
interval_idx = 0

# Temporäre Variablen zum Aufsummieren der kodierten Werte innerhalb eines Intervalls
time_sum = 0
x_sum = 0
y_sum = 0

# Schleife über alle Zeitstempel
for idx in range(len(time_noise)):
    # Solange der Zeitstempel noch im aktuellen Intervall ist, summiere die Werte auf
    if time[idx] < current_interval_end:
        time_sum += time_encoded_decimal[idx]
        x_sum += x_encoded_decimal[idx]
        y_sum += y_encoded_decimal[idx]
    else:
        # Füge die Summe dem aktuellen Intervall hinzu
        S_SEP[interval_idx] = time_sum + x_sum + y_sum

        # Setze die Summen für das nächste Intervall zurück
        time_sum = time_encoded_decimal[idx]
        x_sum = x_encoded_decimal[idx]
        y_sum = y_encoded_decimal[idx]

        # Update der Intervallgrenzen und Intervall-Index
        current_interval_start = current_interval_end
        current_interval_end += delta_t
        interval_idx += 1

print("S_SEP Calculation finished")
print(np.max(N))
S_SEP = S_SEP[:-1]
R_EVS = ((N * S) + (S * int(np.ceil(delta_t / 64)))) / (delta_t / 1000000)  # EVT 2.0
R_compressed = (N * S_compressed) / (delta_t / 1000000)  # IEP-HUFF
R_SEP = (S_SEP + N)  / (delta_t / 1000000) # HUFF_SEP
R = (N * S_raw) / (delta_t / 1000000) # Raw

# Bitrate in Megabit pro Sekunde umrechnen
R_Mbps = R / 1e6
R_EVS_Mbps = R_EVS / 1e6
R_compressed_Mbps = R_compressed / 1e6
R_SEP_Mbps = R_SEP / 1e6

# Zeit in Sekunden berechnen
time_seconds = np.arange(len(R_Mbps)) / (1e6 / delta_t)

bandwith_sensor = 346 * 260 * 8 * 30 / 1e6

# Erstellung des Plots und der Animation
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, np.max(time_seconds))
ax.set_ylim(0, 1500)
line, = ax.plot([], [], label='EVS.RAW', color='g')
line_EVS, = ax.plot([], [], label='EVS.EVT2', color='m')
line_SEP, = ax.plot([], [], label='HUFF_SEP', color='purple')
line_compressed, = ax.plot([], [], label='IEP-HUFF', color='b')
ax.set_xlabel('Zeit (s)', fontsize=14)
ax.set_ylabel('Bandbreite (Mbit/s)', fontsize=14)
ax.set_title('Bandbreitensimulation', fontsize=16)

# Linien für die verschiedenen Bitraten hinzufügen
ax.axhline(y=220.8, color='y', linestyle='--')
ax.axhline(y=496.8, color='r', linestyle='--')

# Beschriftungen über den Linien hinzufügen
ax.text(x=np.max(time_seconds) * 0.55, y=220.8 + 10, s='720p/30FPS Grayscale', color='y', ha='center', fontsize=14)
ax.text(x=np.max(time_seconds) * 0.55, y=496.8 + 10, s='1080p/30FPS Grayscale', color='r', ha='center', fontsize=14)
ax.grid(True)
# Achsenbeschriftungen vergrößern
ax.tick_params(axis='both', which='major', labelsize=14)
# Info-Text oben rechts hinzufügen
textstr = r'$\Delta t = {} ms$'.format(delta_t/1000)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=14,
    verticalalignment='top', horizontalalignment='right', bbox=props)

# Initialisierungsfunktion für die Animation
def init():
    line.set_data([], [])
    line_compressed.set_data([], [])
    line_SEP.set_data([], [])
    line_EVS.set_data([], [])
    return line, line_compressed, line_SEP, line_EVS

def update(frame):
    x_data = time_seconds[:frame]
    y_data = R_Mbps[:frame]
    y_data_compressed = R_compressed_Mbps[:frame]
    y_data_SEP = R_SEP_Mbps[:frame]
    y_data_EVS = R_EVS_Mbps[:frame]
    
    # Setze die neuen Daten für alle drei Linien
    line.set_data(x_data, y_data)
    line_compressed.set_data(x_data, y_data_compressed)
    line_SEP.set_data(x_data, y_data_SEP)
    line_EVS.set_data(x_data, y_data_EVS)
    
    # Rückgabe aller Linien, um die Animation zu aktualisieren
    return line, line_compressed, line_SEP, line_EVS

# Berechne die Anzahl der Frames für die Animation
frames_per_second = 1 / (delta_t / 1e6)  # Wie viele Frames pro Sekunde für die Daten
plt.legend(loc='upper center')
if platform.system() == "Windows":
    plt.rcParams['animation.ffmpeg_path'] = r'ffmpeg.exe' # Pfad zu ffmpeg.exe
# Animation erstellen
ani = FuncAnimation(fig, update, frames=len(time_seconds), init_func=init, blit=True, interval=1000 / frames_per_second, repeat=False)
ani.save('bitrate_animation.mp4', writer='ffmpeg', fps=frames_per_second)
plt.savefig('bitrate_plot.png', dpi=300)
plt.savefig('bitrate_plot_transparent.png', dpi=300, transparent=True)
plt.show()
