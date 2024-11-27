# Projektbeschreibung

Dieses Projekt führt eine Datenkompressionsanalyse mit Huffman-Codierung von Prädiktorabweichungen auf ereignisbasierten Sensordaten durch und visualisiert die benötigte Bandbreite für die Übertragung dieser Daten.

## Dateien

### Model.py

Diese Datei führt die Hauptanalyse der Datenkompression durch. Die wichtigsten Schritte sind:

1. **Daten einlesen**: Liest Ereignisdaten aus einer HDF5-Datei.
2. **Rauschen hinzufügen**: Fügt den Daten basierend auf angegebenen Genauigkeiten Rauschen hinzu.
3. **RMSE berechnen**: Berechnet den Root Mean Square Error (RMSE) für x, y und Zeit.
4. **Huffman-Codierung**: Codiert die Delta-Werte von x, y und Zeit mit Huffman-Codierung.
5. **Kompressionsraten berechnen**: Berechnet Kompressionsraten für verschiedene Genauigkeiten.
6. **Ergebnisse visualisieren**: Stellt die Kompressionsraten gegen die Genauigkeiten grafisch dar.

Die Datei erstellt und speichert Plots, die die Kompressionsraten in Abhängigkeit von der Genauigkeit zeigen.

Die von `Model.py` erzeugten `.png` Dateien sind:

- `compression_rate_vs_accuracy.png`: Zeigt die Kompressionsraten in Abhängigkeit von der Genauigkeit.
- `compression_rate_vs_accuracy_transparent`: Zeigt das gleiche wie compression_rate_vs_accuracy.png nur mit transparentem Hintergrund

Diese Plots helfen dabei, die Effizienz der Huffman-Codierung und den Einfluss der Genauigkeit auf die Kompressionsrate und den Fehler zu visualisieren.

### BoundedBitRate Animated.py

Diese Datei visualisiert die benötigte Bandbreite für die Übertragung von Ereignisdaten und vergleicht verschiedene Komprimierungsmethoden. Die wichtigsten Schritte sind:

1. **Daten einlesen**: Liest Ereignisdaten aus einer HDF5-Datei.
2. **Huffman-Codierung**: Führt eine Huffman-Codierung auf den Daten durch.
3. **Bitraten berechnen**: Berechnet die Bitrate für verschiedene Komprimierungsmethoden.
4. **Animation erstellen**: Erstellt eine Animation der Bitraten über die Zeit.
5. **Huffman-Baum visualisieren**: Visualisiert den Huffman-Baum basierend auf den gegebenen Codes.

Die Datei erstellt und speichert Plots und Animationen, die die Bitraten der verschiedenen Komprimierungsmethoden über die Zeit darstellen.

Die von `BoundedBitRate Animated.py` erzeugten Dateien sind:

- `bitrate_plot.png`: Zeigt die Bitraten der verschiedenen Komprimierungsmethoden.
- `bitrate_plot_transparent.png`: Zeigt das gleiche wie bitrate_plot.png nur mit transparentem Hintergrund.
- `bitrate_animation.mp4`: Zeigt eine Animation der Bitraten über die Zeit

Diese Dateien helfen dabei, die Effizienz der verschiedenen Komprimierungsmethoden und deren Einfluss auf die benötigte Bandbreite zu visualisieren.

## Anforderungen

- Python 3.x
- h5py
- matplotlib
- numpy
- scipy
- pandas
- graphviz

## Ausführung

Um die Skripte auszuführen, stellen Sie sicher, dass alle Abhängigkeiten installiert sind und führen Sie die Skripte, je nachdem was sie erzeugen wollen aus:

1. `Model.py`
2. `BoundedBitRate Animated.py`

Die Ergebnisse werden als Plots und Animationen im aktuellen Verzeichnis gespeichert.

### Zusätzliche Anforderungen

Für die Ausführung der Skripte sind spezifische Datensätze erforderlich:

- Für `BoundedBitRate Animated.py` wird die Datei `driving_sample.hdf5` von Prophesee benötigt, die hier zu finden ist: [driving_sample.hdf5](https://docs.prophesee.ai/stable/datasets.html#chapter-datasets).
- Für `Model.py` wird die Datei `rec1487857941.hdf5` des DDD20-Datensatzes benötigt (sofern man die Grafik aus dem der Bachelorthesis rekonstruieren möchte), die mit den DDD20-Utils von GitHub exportiert wurde. Es können jedoch auch alle anderen Datensätze aus dem DDD20-Datensatz verwendet werden.
