import string
import unidecode

# Abrimos archivos para leer y donde escribir los datos normalizados
# Es importante abrirlos en encodeamiento uft-8 para normalizarlos correctamente
file = open("references.txt", "r", encoding="utf-8")
resFile = open("references_normalize.txt", "a", encoding="utf-8")

# Leemos el archivo
lines = file.readlines()

# Creamos una tabla que servira para quitar los signos de puntuacion
translator = str.maketrans('', '', string.punctuation) 

for line in lines:
    # Quitamos las tildes, signos de puntuacion y mayusculas
    cleanStr = unidecode.unidecode(line).translate(translator).lower()

    # Escribirmos en la linea normalizada en el archivo
    resFile.write(cleanStr)

print("Punctuation marks successfully deleted")

# Cerramos los archivos
file.close()
resFile.close()