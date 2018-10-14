# Datamining-en-assurance

L’objet de ce projet est d’appliquer des méthodes de Machine Learning sur une base assurantielle. La principale difficulté rencontrée a été de trouver une base de données exploitable publique. 

Au final notre choix s’est porté sur des données fournissant les caractéristiques d’automobiles et leur risque assurantiel. Vous trouverez les données aux liens suivants : https://archive.ics.uci.edu/ml/datasets/automobile

Cette base de données comprend trois types de données :
- Les caractéristiques d’une voiture ;
- Sa côte de risque d’assurance ;
- Ses pertes d'utilisation normalisées en comparaison des autres voitures.

La côte de de risque d’assurance représente le surplus de risque par rapport au risque associé au prix. Les voitures reçoivent initialement un symbole de facteur de risque associé à leur prix. Ensuite, si le risque est jugé plus grand (ou moins), ce symbole est ajusté en le déplaçant vers le haut (ou le bas) de l'échelle. Les actuaires appellent ce processus «symboling». Une valeur de +3 indique que l'auto est risquée, -3 qu'elle est probablement assez sûre. Ainsi, dans cette étude nous allons essayer de prédire cette côte de risque d’assurance en fonction des caractéristiques de la voiture.

Dans une première partie, nous allons analyser et retraiter la base données. Puis dans une seconde partie nous modélisons cette côte de risque d’assurance à l’aide des algorithmes suivants : une régression linéaire, un KNN, un SVM, un arbre de décision, une forêt aléatoire et des réseaux de neurones.

Dans le répetoire, vous trouverez le code en R et le rapport.
