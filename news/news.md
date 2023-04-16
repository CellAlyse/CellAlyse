# Release 1.0.1
> 2023-04-15
### Analyse 
Damit eine komplette Blutzählung möglich ist, wurde ein Analysetool implementiert.
Dieses nimmt die Segmentierungsmasken und berechnet wichtige Eigenschaften der Blutzellen.
Das funktioniert, da Überlappung relativ gut erkannt werden.

#### Die wichtigen Parameter

*Fläche und Durchmesser*
> Durch die Fläche und dem Durchmesser können Deformierungen gezählt und ausgewertet werden. Dadurch
> können Sichelzellen leicht gezählt und in relation gestellt werden. Auch kann der Typ einer Anämie bestimmt werden.
> Auch kann der Stadium einer Leukämie bestimmt werden.

*Extensivität und Orientierung*
> Diese Parameter dienen der Populationsbestimmung.

### Metriken
Über dieses Modul können die Metriken, welche im "Analyse"-Modul berechnet werden, auf Leukozyten 
angewendet werden. Durch die Isolierung kann ein Tuple erstellt werden, wo für jeden Zelltypen die
Parameter bestimmt wurden.

### Die neuen Modelle
LISC und BCCD sind schon fertig implementiert. Da die Hyperparameter auf Raabin bestimmt werden,
wird es noch etwas dauernd sein, bis die Modelle fertig sind.

### Floureszenz
Das Cell-U-Net ist auf dem BBBC-Datensatz trainiert worden und erreicht sogar sehr gute 
Ergebnisse. Instanzmasken werden durch die Kombination des Dual-Outputs und dem Interior Expansion
Algorithmus erstellt. Das Modul funktioniert nur auf Bildern, welche auf eine bestimmte Art vorbereitet 
wurden. Testbilder können aus dem BBBC039-Datensatz heruntergeladen werden. 

*Die unbearbeiteten Bilder können über diesen [Link](https://data.broadinstitute.org/bbbc/BBBC039/) heruntergeladen werden.*

*Normalisierte können über diesen [Link](https://data.broadinstitute.org/bbbc/BBBC039/BBBC039_v1_images_normalized.zip) heruntergeladen werden.*

|    Methode | Dice | AJI | DQ | SQ |
|-----------|:----:|:---:|:--:|:--:|
| Cell-U-Net | 96.452 | 90.2| 94.431| 91.67|

> *AJI*  | Aggregated Jaccard Index 
> 
> *DQ* | Detection Quality
> 
> *SQ* | Segmentation Quality

## Die Bugs
Das Wechseln der Module während der Inferenz führt zum Absturz der Website. Das passiert, weil
die Session noch läuft und die Kapazitäten des Servers überschritten werden. Dies führt
zur Abschaltung von Streamlit. Ob das Problem gelöst werden kann, ist noch nicht bekannt.