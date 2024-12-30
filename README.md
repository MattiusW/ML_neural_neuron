# ML_neural_neuron
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
