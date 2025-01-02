# ML_neural_neuron moje przemyślenia. 
Manuale tworzenie sieci neuronowych bez używania bibliotek od zera.

Pierwszy projekt "neutral_network_from_scratch", utworzyłem podążając za 2 godzinnym tutorialem z YT.
Głównie było to przepisywanie wzorów matematycznych na kod w pythonie. Algorytm działa dobrze, ale 
tylko na zbiorze MNIST ropoznaje cyfry od 0-9. 

Drugi projekt "neural_network_strong_pokemon" jest to ten sam algorytm co w pierwszym projekcie, jednak
próbowałem załadować tam swoje dane. Dane pokemonów ze wszystkich generacji wraz ze statystykami. Model
miał prognozować, który pokemon jest najsliniejszy. Oczywiście jest to banalne bo można by po prostu 
zsumować wszystkie statystyki i zwrócić pokemona z największą liczbą. Jednak próbowałem na prostym
przykadzie nauczyć się dostarczać dane do modelu. Jednak nie wiem czy jest za mało danych czy 
algorytm jest niepoprawny ponieważ model zwraca, zawsze pierwszy rekord pokemona po przetasowaniu danych.
Zastanawiam się czy do prognozowania wyników nie lepszy byłby inny algorytm regresji liniowej a sieci
neurownowe bardziej są odpowiednie np. do klasyfikacji obrazów.

Trzeci projekt "nerual_network_image_pokemon" jest to algorytm podobny do pierwszego jednak zrobiony
z pomocą sztucznej inteligencji. Wzory matematyczne oraz funkcje takie jak relu lub propagacja wsteczna
są od kilku lat niezmiennie więc nikt nie napisze innego algorytmu niż ten, który służy już od lat.
Udało mi się znaleść ciekawą biibloteke zdjęć pokemonów i wytrenowałem sieć neuronową rozpoznajacą 
obrazy pokemonów z dokładnością do 94.66%. Model uczył się 700 epok na 1024 hidden layerach. Model
trenował się około półtorej godziny.

Czwarty projekt "neural_network_pokemon" jest to ten sam algorytm, który został użyty w trzecim projekcie
jednak został uporządkowany zgodnie z konwencjcą obiektowości. Cały algorytm został podzielony na klasy
dzięki czemu cały kod jest bardziej przejrzysty. Jednak trenowałem model prawie całą noc na 1000 epokach
i 1024 ukrytych layerach i uzyskałem wynik tylko 45.26% co bardzo mnie dziwi, gdyż zwiększyłem mu tylko
liczbę epok oraz ilość próbek treningowych była troszke większa niż w trzecim projekcie a mimo to model
uczył się o wiele dłużej i do tego jeszcze nie jest tak dokładny jak trzeci. Nie wiem co może być 
tego przyczyną.
