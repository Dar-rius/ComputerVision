L'idée du projet est d’utiliser un modèle de réseau de neurone pré-entraîner d’OpenCV qui pourra reconnaître les visages des étudiants de la 
LPIG2 (Licence 2 informatique et gestion)  d’UCAO en affichant leur noms sur l'écran, j’ai donc créé un jeu de donnée dans un dossier nommé “Images” qui contient différents 
dossiers de chaque étudiants avec leur photo à l'intérieur. Pour que le modèle puisse identifier les différents labels, les labels correspondent aux noms des différents 
dossiers contenus dans le dossier Image qui eux sont égale aux noms des différents étudiants. Suite à ça le modèle va enregistrer les données encodées dans un fichier 
nommé “face_enc” qui sera utilisé lors de la détection du  visage pour savoir si ce visage est proche de l'un des visage enregistrer dans notre jeu de donnée, si oui alors 
il affichera son nom à l'écran si non il marquera qu’il est inconnu.  Les différentes librairies utiliser pour le projet: face-recognition, DLIB et OpenCV.

Pour Essayer le projet dans votre ordinateur:

Installer opencv:

``pip install opencv-python``

Installer dlib: [DLIB](http://dlib.net/)

Installer face-recognition: 

``pip install face-recognition``

Puis faites un git clone et lancer le programme.
