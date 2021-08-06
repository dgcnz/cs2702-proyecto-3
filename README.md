# Proyecto 3 - Base de Datos Multimedia

## Datos generales
- `Curso:` Base de Datos III - cs2702.
- `Profesor:` Heider Sanchez.

### Integrantes
- Andrea Díaz
- Diego Cánez
- Maor Roizman

## Descripción
Este proyecto está enfocado a la construcción óptima de una estructura multidimensional para dar soporte a las búsqueda y recuperación eficiente de imágenes en un servicio web de reconocimiento facial.

### Objetivo del Proyecto
El logro del estudiante está enfocado a entender y aplicar los algoritmos de búsqueda y recuperación de la información basado en el contenido.

### Frontend

![Index](https://github.com/dgcnz/cs2702-proyecto-3/blob/master/images/index.png?raw=true)
![Index With Message](https://github.com/dgcnz/cs2702-proyecto-3/blob/master/images/index-with-messsage.png?raw=true)
![Galery](https://github.com/dgcnz/cs2702-proyecto-3/blob/master/images/galery.png?raw=true)


### Construcción del índice RTree

### Algoritmo de búsqueda KNN

### Algoritmo de búsqueda por Rango

### Análisis y discusión de la experimentación.
![Benchmarks](https://github.com/dgcnz/cs2702-proyecto-3/blob/master/images/tests.png?raw=true)
|           | KNN - RTree | KNN - Secuencial |
|-----------|-------------|------------------|
| N = 100   |    0.0002   |      0.0003      |
| N = 200   |    0.0003   |      0.0006      |
| N = 400   |    0.0005   |      0.0011      |
| N = 800   |    0.001    |      0.0023      |
| N = 1600  |    0.0019   |      0.0045      |
| N = 3200  |    0.0041   |      0.0023      |
| N = 6400  |    0.0092   |      0.0184      |
| N = 12800 |    0.0191   |      0.0378      |

### Video de presentación
