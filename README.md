# Práctica 4 de VC (Visión por Computador)

## Autores

Leslie Liu Romero Martín
<br>
María Cabrera Vérgez

## Tareas realizadas

Para la entrega de esta práctica, la tarea consiste en desarrollar un prototipo que procese uno (vídeo ejemplo proporcionado o varios vídeos (incluyendo vídeos de cosecha propia)):

- detecte y siga las personas y vehículos presentes
- detecte las matrículas de los vehículos presentes
- cuente el total de cada clase
- vuelque a disco un vídeo que visualice los resultados
- genere un archivo csv con el resultado de la detección y seguimiento. Se sugiere un formato con al menos los siguientes campos:
	* fotograma
	* tipo_objeto
	* confianza
	* identificador_tracking
	* x1
	* y1
	* x2
	* y2
	* matrícula_en_su_caso
	* confianza
	* mx1
	* my1
	* mx2
	* my2
	* texto_matricula

## Instalación

``` python
import cv2  
import math 

from ultralytics import YOLO
from matplotlib import pyplot as plt
import easyocr
import pandas as pd
import csv
```
## Tareas

### OCR

Para leer el texto de las placas, uno de los modelos usados ha sido EasyOCR, un modelo de python para extraer texto de imágenes (en este caso, de los frames de un vídeo). Para poder analizar cómodamente el vídeo, se generan unas imágenes recortadas, obtenidas a partir de las coordenadas del bouding box.

``` python
	    x1m = max(0, x1 - margin)
            y1m = max(0, y1 - margin)
            x2m = min(frame.shape[1], x2 + margin)
            y2m = min(frame.shape[0], y2 + margin)
            placa_crop = frame[y1m:y2m, x1m:x2m]
```

Si se ha generado una imagen recortada, si esta tiene píxeles, y por lo tanto no está vacía, se aumentará el tamaño de la imagen para que el OCR intente leerlo y se va a procurar que tenga buena calidad a pesar de la redimensión.

``` python
escala = 3
placa_crop = cv2.resize(placa_crop, None, fx=escala, fy=escala, interpolation=cv2.INTER_CUBIC)
```

< inserte imagen con la redimensión, ejemplo de crop >

A continuación y con el objetivo de simplificar el futuro procesamiento de la imagen para su lectura, se aplican algunos filtros. Entre ellos está convertir la imagen de una escala BGR (blue, green, red) a gris. Se aplica una ecualización de histograma a la imagen generada. El motivo de ello es mejorar el contraste de la imagen, facilitar el trabajo del OCR y que no confunda el fondo con el texto a leer.

``` python
gray = cv2.cvtColor(placa_crop, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
```

Se procede a aplicar el OCR en la imagen con el filtro gris. Los otros parámetros que se le pasan son los caracteres permitidos dentro de una matrículas (todo el abecedario en mayúsculas y los número del 0-9). Por último, detail = 1 indica que todos los elementos de la lista de resultados tendrá la forma [coordenadas, texto, confianza].

``` python
ocr_result = reader.readtext(

                    gray,

                    allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',

                    detail=1

                )
```

Si se ha generado algún resultado (porque puede pasar que al inicio no se detecte ninguna matrícula), se guardará el texto por un lado, eliminando los espacios en blancos por medio de .strip(). En una variable llamada "prob" se guardará la confianza del OCR. Se toma ocr_result[0] porque será la matrícula principal en ese momento.

``` python
if len(ocr_result) > 0:

                    text = ocr_result[0][1].strip()

                    prob = ocr_result[0][2]
```

Puede pasar que el OCR genere matrículas donde solo detecta una letra. Para ello, se le ponen ciertas condiciones. Por un lado, que la matrícula debe tener como poco 4 caracteres. La probabilidad debe ser mayor al 50 % (0'5) y que el texto a analizar sea diferente al último guardado (por tratar de evitar guardar la misma matrículas varias veces por los diversos frames).

Se obtiene el tiempo en segundos del frame que captó la matrícula y se imprime, indicando su confianza. Además, se ilustra encima del propio vídeo, colocando por el tiempo que permita la matrícula dentro del vídeo. Se sitúa, indica la fuente, el tamaño, el color, grosor.

``` python
if len(ocr_result) > 0:
                    text = ocr_result[0][1].strip()
                    prob = ocr_result[0][2]

                    if len(text) >= 4 and prob > 0.5 and text != last_plate:
                        last_plate = text
                        timestamp = vid.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        print(f"[{timestamp:.2f}s] Matrícula: {text} (Conf: {prob:.2f})")
                        cv2.putText(frame, f'{text}', (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Detección + OCR", frame)
```

### Guardar datos en csv

Para guardar los datos dentro de un csv, se ha hecho uso de la función del mismo nombre. Se pasa una lista de listas con los siguientes datos formando las columnas: 

* fotograma: frame del vídeo
* tipo_objeto: clase del objeto detectado 
* confianza: confianza de la detección (entre 0 y 1)
* id_tracking: id
* x1: coordenada del bounding box
* y1: coordenada del bounding box
* x2: coordenada del bounding box
* y2: coordenada del bounding box
* matrícula_en_su_caso: Indica si se detectó una matrícula
* conf_ocr: confianza del ocr
* mx1: coordenada del bounding box de la matrícula
* my1: coordenada del bounding box de la matrícula
* mx2: coordenada del bounding box de la matrícula
* my2: coordenada del bounding box de la matrícula
* texto_matricula: texto captado

``` python
columnas = [
        "fotograma", "tipo_objeto", "confianza", "id_tracking",
        "x1", "y1", "x2", "y2",
        "matricula_detectada", "conf_ocr",
        "mx1", "my1", "mx2", "my2",
        "texto_matricula"
    ]
```

Se abre el archivo en modo escritura (resultado.csv) para poner añadir los nuevos datos. Se va escribiendo por filas, usando ';' como separador, esto debido a que Excel reconoce así la separación entre columnas. Se escriben los encabezados y luego los datos. Por último, se envía un mensaje avisando al usuario de que el archivo ya ha sido generado.
