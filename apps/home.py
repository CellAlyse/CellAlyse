import streamlit as st



def home():
    st.markdown("# Was ist CellAlyse")
    st.markdown("Mit CellAlyse können Blutzellen mit leichtigkeit gezählt werden. Die Automatisierung der Zählung spart Zeit, reduziert Fehler und ermöglicht eine bessere Diagnose.")

    st.markdown("## Weiße Blutzellen")
    st.markdown("Mit diesem Modul kann man einzelne Zellen identifizieren und klassifizieren. Durch das Klassifizieren einer Vielzahl von Zellen kann man schließlich die Gesamtzahl jeder Art von Zellen im Bild ermitteln. Das Modul erleichtert die Überwachung und Analyse von Blutzellen und kann somit bei der Diagnose und Überwachung von Blutkrankheiten hilfreich sein.")

    st.image("storage/images/media/WBC.gif")

    st.markdown("### Bounding Box")
    st.markdown("Die Bounding Box ist eine rechteckige Grenze, die eine Zelle einschließt. Mit ihr kann man die Anzahl und die Lage der weißen Blutzellen einfach bestimmen. Dies ist eine wichtige Information für die weitere Analyse und Klassifizierung der Zellen. Die Bounding Box hilft, den Bereich der Zelle zu identifizieren, der für weitere Analysen relevant ist, und reduziert so den Rechenaufwand und die Fehleranfälligkeit.")

    st.markdown("### Klassifizierung")
    st.markdown("In der Klassifizierung gibt es mehrere verfügbare Modelle, die man auswählen kann. Diese Modelle sind jeweils für einen spezifischen Typ von Mikroskop und ein bestimmtes Vorbereitungsverfahren optimiert. Es ist wichtig, das richtige Modell auszuwählen, um die bestmöglichen Ergebnisse zu erzielen und Fehler zu vermeiden. Es empfiehlt sich, die Verfügbarkeit und die Anpassbarkeit der Modelle sorgfältig zu überprüfen, um sicherzustellen, dass das ausgewählte Modell den Anforderungen entspricht und die gewünschten Ergebnisse liefert.")

    st.markdown("---") 

    st.markdown("## Typisierung")
    st.markdown("Dieses Modul nutzt deep learning, um die Anzahl von roten Blutzellen, weißen Blutzellen und Blutplättchen in einem Bild zu erkennen und zu zählen. Da es sich um ein komplexes Modell handelt, kann die Verarbeitung einige Zeit in Anspruch nehmen, bis die Ergebnisse bereitgestellt werden.")

    st.image("storage/images/media/RBC.gif")

    st.markdown("### Circle Hough Transform")
    st.markdown("Mit dem Circle Hough Transform Algorithmus kann man Kreise in einem Bild erkennen. Dieser Algorithmus eignet sich besonders für die Analyse von roten Blutzellen, die in der Regel eine runde Form aufweisen. Um auch kleine oder große Kreise erkennen zu können, kann man die Minimal- und Maximalradien anpassen. Auf diese Weise kann man sicherstellen, dass der Algorithmus alle relevanten Kreise erkennt und zählt.")

    st.markdown("### Component Labeling")
    st.markdown("Der Component Labeling Algorithmus ist eine Methode, die verbundene Teilmengen identifizieren und zählen kann. Dieser Algorithmus eignet sich hervorragend für die Analyse von Blutplättchen, da diese oft isoliert und mit sehr kleinen Flächen vorliegen. Mit diesem Verfahren kann man die Anzahl der Plättchen schnell und effektiv ermitteln.")

    st.markdown("### Distance Transform")
    st.markdown("Der Distance Transform Algorithmus verwendet Distanzinformationen, um zusammenhängende Objekte voneinander zu unterscheiden. Diese Methode ist besonders effektiv bei der Analyse von weißen Blutzellen, da diese oft zusammenhängend, aber nicht überlappend sind.")

    st.markdown("---")
    
    st.markdown("## Datensatz")
    st.markdown("Der Datensatz umfasst 75 Bilder von mittelmäßiger Qualität. Zu Beginn wird ein Graph angezeigt, der die Verteilung visualisiert. Am Ende kann man die Ergebnisse der KI mit den tatsächlichen Ergebnissen vergleichen, um die Genauigkeit zu bewerten.")

    st.image("storage/images/media/dataset.gif")

