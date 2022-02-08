from imutils import paths
import face_recognition
import pickle
import cv2
import os
 

#Obtenir les paths de chaque fichier depuis le dossier Image qui
#l'equivalent d'un dataset
imagePaths = list(paths.list_images('Images'))
knownEncodings = []
knownNames = []

# Boucle pour recuperer les paths des files
for (i, imagePath) in enumerate(imagePaths):
    # extraire les labels depuis le nom de leur dossier
    name = imagePath.split(os.path.sep)[-2]
    # charger les images en entree et les convertir en BGR (OpenCV ordering)
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #Utiliser face_recognition pour localiser les visages
    boxes = face_recognition.face_locations(rgb,model='hog')
    # encoder les visage
    encodings = face_recognition.face_encodings(rgb, boxes)
    # une boucle pour l'encodage des visages
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
#Enregistrer les visages encoder et les labels dans un dictionnaire
data = {"encodings": knownEncodings, "names": knownNames}
#Utiliser pickle pour enregistrer les donnees un file pour l'utiliser plus
#tard lors de la detection de l'apppartenance du visage
f = open("face_enc", "wb")
f.write(pickle.dumps(data))
f.close()



#Trouver le path du fichier xml contenu dans le fichier haarcascade 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# charger les harcaascade dans le cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
# charger les donnees donnees du file "face_enc"
data = pickle.loads(open('face_enc', "rb").read())
 
print("Streaming started")
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
 
    # convertir les entree BRG en RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # encoder les donnees en entreee
    encodings = face_recognition.face_encodings(rgb)
    names = []

    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],
         encoding)
        #set name =Inconnu if no encoding matches
        name = "Inconnu"
        # check to see if we have found a match
        if True in matches:
            #Find positions at which we get True and store them
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matchedIdxs:
                #Check the names at respective indexes we stored in matchedIdxs
                name = data["names"][i]
                #increase count for the name we got
                counts[name] = counts.get(name, 0) + 1
            #set name which has highest count
            name = max(counts, key=counts.get)
 
 
        # update the list of names
        names.append(name)
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()