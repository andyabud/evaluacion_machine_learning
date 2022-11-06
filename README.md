# Evaluación Machine Learning
Código usado para resolver la evaluación de Machine Learning

En este ejercicio se crean distintos modelos supervisados: SVM, Decision Tree y Random Forest. Basados en una base de datos meteorológica donde la variable de respuesta es si llueve mañana o no (LluviaMan)

Se realizan una serie de pasos para trabajar con los datos, como por ejemplo:
- Escalado y normalización de datos
- Uso de variables dummy
- Creación de base de entrenamiento y testeo a partir de la base estandarizada, con una proporción 70% - 30%
- Uso del framework Tidymodels
- Elección del mejor modelo en base a la métrica ROC_AUC

La rúbrica de esta evaluación es la siguiente


Parte 1 - Modelos supervisados

En un centro meteorológico, se contratan sus servicios como Data Scientist para construir un modelo que prediga si lloverá o no en las próximas 24 horas, utilizando información de las 24 horas previas. Se dispone de un conjunto de 19 variables meteorológicas. La descripción de las variables es la siguiente:

    MinTemp Temperatura mínima registrada.
    MaxTemp Temperatura máxima registrada.
    Lluvia Cantidad de lluvia registrada ese día en mm.
    Evaporación Evaporación (mm) en 24 horas.
    Sol Número de horas de sol brillante en el día.
    VelRafaga La velocidad (km/h) de la ráfaga de viento más fuerte en las 24 horas hasta la medianoche.
    Vel9am La velocidad (km/h) de la ráfaga de viento a las 9am.
    Vel3pm La velocidad (km/h) de la ráfaga de viento a las 3pm.
    Hum9am Porcentaje de humedad a las 9am.
    Hum3pm Porcentaje de humedad a las 3pm.
    Pres9am Presión atmosférica (hpa) a nivel del mar a las 9am.
    Pre3pm Presión atmosférica (hpa) a nivel del mar a las 3pm.
    Nub9am Fracción del cielo cubierto por nubes a las 9am. Se mide en “octavos”, de manera que un valor 0 indica cielo totalmente despejado y 8, cielo totalmente cubierto.
    Nub3pm Fracción del cielo cubierto por nubes a las 3pm. Se mide en “octavos”, de manera que un valor 0 indica cielo totalmente despejado y 8, cielo totalmente cubierto.
    Temp9am Temperatura en grados celsius a las 9am.
    Temp3pm Temperatura en grados celsius a las 3pm.
    LluviaHoy Variable indicadora que toma el valor 1 si la precipitación en mm. en las últimas 24 horas excede 1 mm. y 0 si no.
    Koppen Clasificación Koppen de la zona de medición (Temperate, Subtropical, Grassland, Tropical, Desert).
    Estación Estación del Año.
    LluviaMan Indicador de lluvia al día siguiente de la medición.

Pregunta 1

Utilizando la base de datos Lluvia_ML.csv, realice una separación de la base de datos en un set de entrenamiento y set de validación. Utilice una proporción de 70:30 respectivamente estratificado por la LluviaMan. Para poder replicar sus resultados, fije una semilla antes de obtener los indices. Para ello utilice la función set.seed(2022).

Nota: Una base separada estratificadamente quiere decir que la proporción de casos positivos como negativos es equivalente en ambas muestras.

    Puntaje: 0.5p

Pregunta 2

Utilizando el set de entrenamiento, ajuste un modelo de SVM, árbol de clasificación y Random Forest. Para ello debe:

    Limpiar los datos:
        Escalar los datos numéricos.
        Transformar a dummys las variables categóricas.

    Definir la grilla de hiperparámetros a optimizar, en este caso los hiperparámetros a optimizar son:
        SVM
            Costo
        Árbol de clasificación:
            Observaciones mínimas para división.
            Costo de complejidad.
            Profundidad máxima del árbol.
        Random Forest
            Número de árboles.
            Número de predictores muestreados para cada árbol.
            Observaciones mínimas para división.

    Buscar los hiperparámetros óptimos utilizando validación cruzada optimizando la métrica de roc_auc

    Puntaje: 2p

Pregunta 3

Compare los modelos ajustados utilizando el set de test, bajo el criterio de roc_auc, ¿cuál es el mejor modelo en su caso?

    Puntaje 0.5p

Parte 2 - Modelos no supervizados

El set de datos datos clustering.csv contiene 1000 observaciones simuladas, en la que se poseen 10 atributos numéricos llamadas V1 a V10, además, del atributo y que indica el grupo real de la observación.
Pregunta 4

Realice un análisis de componentes principales, seleccione la cantidad de componentes que expliquen al menos el 70% de la variabilidad de los datos.

Nota: Importante no considerar la variable y en el estudio de PCA, ni en la creación de los segmentos.

    Puntaje 1p

Pregunta 5

Utilizando la set de datos construidos con los PCA en el paso anterior, determine el número optimo de clustering para una segmentación k-means, utilizando la metodología de Elbow y Average silhouette.

    Puntaje 1p

Pregunta 6

    Ajuste un clustering por medio de la metodología K-means usando el número de centros encontrados en el paso anterior

    Utilizando las 2 principales componentes principales, gráfique los segmentos ajustados, compárelo con el gráfico de los segmentos reales de los datos.

¿La metodología aplicada en este caso es apropiada?

    Puntaje 1p

