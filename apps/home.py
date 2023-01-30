import streamlit as st


def home():
    st.markdown("# CellAlyse")

    st.markdown("## Was ist CellAlyse?")
    st.markdown(
        "CellAlyse ist eine Webanwendung, die es ermöglicht, Blutausstriche auszuwerten."
        "Die Auswerung erfolgt mittels Machine Learning und Computer Vision."
    )

    st.markdown("## Was kann ich machen?")
    st.markdown(
        "Generell gibt es zwei seperate KIs. Die eine kann weiße Blutzellen klassifizieren und zählen."
        "Die andere kann verschiedene Blutzelltypen zählen. Das einzige was man braucht ist ein Bild von "
        "einem Blutausstrich."
    )

    st.markdown("## Wie kann ich weiße Blutzellen klassifizieren?")
    st.markdown(
        "Auf der linken Seite befinden sich die Module der Website."
        'Die KI kann unter dem Modul "weiße Blutzellen" ausprobiert werden.'
        "Dort können die Nuklei von weißen Blutzellen segmentiert werden und die Zellen können natürlich "
        "klassifiziert werden."
    )

    st.markdown("## Wie kann ich Blutzellen zählen?")
    st.markdown(
        'Unter dem Modul "weiße Blutzellen" befindet sich das Zählungsmodul.'
        "Hier kann man auf der linken Seite verschiedene Einstellungen verändern."
        "Zunächst kann man den Typen der Blutzelle einstellen, dannach können die verschiedenen Zählalgorithmen"
        "ausgewählt werden. Für rote Blutzellen empfehlen wir Circle Hough Transform zu verwenden. Bei weißen"
        "Blutzellen und Plättchen empfehlen wir connected component labeling und distance transform zu verwenden."
        "Wenn man eine andere Vergrößerung oder Auflösung hat, können die Parameter angepasst werden."
    )

    st.markdown("## Ich habe einen Datensatz, was jetzt?")
    st.markdown("Da wir in der Online Version keine batch uploads emp")
