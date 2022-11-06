
# Cargar Librerías --------------------------------------------------------

library(tidyverse)
library(Hmisc)
library(tidymodels)
library(readr)
library(randomForest)
library(caret)
library(ROSE)
library(PRROC)
library(ISLR)
library(rpart)
library(rpart.plot)
library(rattle)
library(fastDummies)
library(e1071)
library(kernlab)
library(factoextra)


# 1. Cargar base y prepararla ---------------------------------------------

## Cargar dataframe
lluvia <- read.delim("data/lluvia_ml.csv", sep = ",")

### Fijar semilla 2022
set.seed(2022)

## Hacer un initial split estratificado con la función initial_split
lluvia_strat <- rsample::initial_split(data = lluvia,
                              strata = LluviaMan,
                              prop = 0.7)

#Obtener dataframe de entrenamiento
lluvia_train <- rsample::training(lluvia_strat)

# Obtener dataframe de testeo
lluvia_test <- rsample::testing(lluvia_strat)


# 2. Ajuste de modelos ----------------------------------------------------

## 2.1.a Limpiar los datos - Escalar datos numéricos----

### Hacer un glimpse a la base lluvia para identificar las variables numéricas
glimpse(lluvia)

### Escoger las variables numéricas
example_set <- lluvia %>% dplyr::select(MinTemp, MaxTemp, Lluvia, Evaporacion, Sol, 
                                 VelRafaga, Vel9am, Vel3pm, Hum9am, Hum3pm,
                                 Pres9am, Pre3pm, Nub9am, Nub3pm, Temp9am,
                                 Temp3pm)

### Escalar los datos numéricos con preProcess
preprocess_data <- caret::preProcess(example_set, method=c("range"))

### Estandarización de datos
process_data <- predict(preprocess_data, example_set)

### Crear base para pegar valores estandarizados
lluvia_std <- example_set

### Pegar variables estandarizadas en la base recién creada
lluvia_std$MinTemp <- process_data$MinTemp
lluvia_std$MaxTemp <- process_data$MaxTemp
lluvia_std$Lluvia <- process_data$Lluvia
lluvia_std$Evaporacion <- process_data$Evaporacion
lluvia_std$Sol <- process_data$Sol
lluvia_std$VelRafaga <- process_data$VelRafaga
lluvia_std$Vel9am <- process_data$Vel9am
lluvia_std$Vel3pm <- process_data$Vel3pm
lluvia_std$Hum9am <- process_data$Hum9am
lluvia_std$Hum3pm <- process_data$Hum3pm
lluvia_std$Pres9am <- process_data$Pres9am
lluvia_std$Pre3pm<- process_data$Pre3pm
lluvia_std$Nub9am<- process_data$Nub9am
lluvia_std$Nub3pm <- process_data$Nub3pm
lluvia_std$Temp9am <- process_data$Temp9am
lluvia_std$Temp3pm <- process_data$Temp3pm

### Agregar las variables categóricas de la base inicial
lluvia_std$LluviaHoy <- as.factor(lluvia$LluviaHoy)
lluvia_std$LluviaMan <- as.factor(lluvia$LluviaMan)
lluvia_std$Koppen <- as.factor(lluvia$Koppen)
lluvia_std$Estacion <- as.factor(lluvia$Estacion)

### Chequear que todo esté bien con un glimpse
glimpse(lluvia_std) # Todas las variables son del tipo correcto

## 2.1.b Crear variables dummy----
lluvia_std_dummy <- lluvia_std
lluvia_std_dummy <- fastDummies::dummy_cols(lluvia_std_dummy, 
                                            select_columns = c("LluviaHoy",
                                                               "LluviaMan",
                                                               "Koppen",
                                                               "Estacion"))

### Eliminar las variables categóricas originales ahora que tenemos las dummy, para evitar colinealidad
lluvia_std_dummy_ok <- lluvia_std_dummy %>% 
  dplyr::select(-LluviaHoy, -LluviaMan, -Koppen, -Estacion, -LluviaHoy_No, -LluviaMan_No)

### Transformar variables dummy a factor
glimpse(lluvia_std_dummy_ok)
lluvia_std_dummy_ok$LluviaHoy_Yes <- as.factor(lluvia_std_dummy_ok$LluviaHoy_Yes)
lluvia_std_dummy_ok$LluviaMan_Yes <- as.factor(lluvia_std_dummy_ok$LluviaMan_Yes)
lluvia_std_dummy_ok$Koppen_Desert <- as.factor(lluvia_std_dummy_ok$Koppen_Desert)
lluvia_std_dummy_ok$Koppen_Grassland <- as.factor(lluvia_std_dummy_ok$Koppen_Grassland)
lluvia_std_dummy_ok$Koppen_Subtropical <- as.factor(lluvia_std_dummy_ok$Koppen_Subtropical)
lluvia_std_dummy_ok$Koppen_Temperate <- as.factor(lluvia_std_dummy_ok$Koppen_Temperate)
lluvia_std_dummy_ok$Estacion_Invierno <- as.factor(lluvia_std_dummy_ok$Estacion_Invierno)
lluvia_std_dummy_ok$Estacion_Otoño <- as.factor(lluvia_std_dummy_ok$Estacion_Otoño)
lluvia_std_dummy_ok$Estacion_Primavera <- as.factor(lluvia_std_dummy_ok$Estacion_Primavera)
lluvia_std_dummy_ok$Estacion_Verano <- as.factor(lluvia_std_dummy_ok$Estacion_Verano)

### Crear base de entrenamiento y testeo a partir de la base estandarizada
set.seed(2022)
lluvia_split_std <- rsample::initial_split(data = lluvia_std_dummy_ok,
                                           strata = LluviaMan_Yes,
                                           prop = 0.7)

#Obtener dataframe de entrenamiento
lluvia_train_std <- training(lluvia_split_std)
#Obtener dataframe de testeo
lluvia_test_std <- testing(lluvia_split_std)


## 2.2 Optimizar hiperparámetros----

### Definir datos a ocupar
datos_procesados <- recipes::recipe(formula = LluviaMan_Yes ~ .,
                                    data = lluvia_std_dummy_ok)

### Definir folds
set.seed(2022)
crossv <- vfold_cv(data = lluvia_train_std,
                   v = 3,    # cantidad de folds
                   strata = LluviaMan_Yes)

## 2.2.a Support Vector Machine----

### Definir tipo de modelo----
mod_svm <- parsnip::svm_linear(mode = "classification",
                      cost =  tune()) %>% # parametros a buscar en grilla
  parsnip::set_engine("kernlab") 

### Definir workflow----

### Desde acá se ajusta el modelo
w_svm <- workflows::workflow() %>% 
  add_recipe(datos_procesados) %>%
  add_model(mod_svm)

### Grilla de hiperparámetros----
hp_svm <- grid_regular(cost(range = c(-2, 4)),
                       levels= c(30))

### Búsqueda de hiperparámetros----
grilla_svm <- tune_grid(object = w_svm, # El workflow que acabamos de crear
                        resamples = crossv, # La validación cruzada que creamos
                        metrics = metric_set(roc_auc), # Métrica que se ocupará para evaluar
                        control = control_resamples(save_pred = TRUE), #Guardar predicciones evaluadas
                        grid= hp_svm) # La grilla que ocuparemos


### Definir mejor modelo----
svm_best <- finalize_workflow(x= w_svm, 
                              parameters= select_best(grilla_svm, metric= "roc_auc")) %>% 
  fit(lluvia_train_std)

### Print modelo
svm_best


## 2.2.b Arbol de Decisión -----------------------------------------------

### Definir tipo de modelo----
mod_arbol <- parsnip::decision_tree(mode = "classification",
                           cost_complexity = tune(), #Costo de complejidad
                           tree_depth = tune()) %>%  #Profundidad del árbol 
  set_engine("rpart", parms= list(split= "information"))

### Definir workflow----
w_arbol <- workflow() %>% 
  add_recipe(datos_procesados) %>% 
  add_model(mod_arbol)

### Grilla de hiperparámetros----
hp_arbol <- grid_regular(cost_complexity(range = c(0, 1), trans = NULL),
                         tree_depth(range = c(3, 8), trans = NULL),
                         levels= c(8, 3))

### Búsqueda de hiperparámetros----
grilla_arbol <- tune_grid(object = w_arbol,
                          resamples = crossv,
                          metrics = metric_set(roc_auc),
                          control = control_resamples(save_pred = TRUE),
                          grid= hp_arbol)

### Definir mejor modelo----
dt_best <- finalize_workflow(x= w_arbol, 
                              parameters= select_best(grilla_arbol, metric= "roc_auc")) %>% 
  fit(lluvia_train_std)

### Print modelo
dt_best


# 2.2.c Random Forest -----------------------------------------------------

### Definir tipo de modelo----
mod_rf <- rand_forest(mode = "classification", 
                      mtry  = tune(), # Cantidad de predictores
                      trees = tune()) %>%  # Cantidad de arboles
  set_engine("ranger", parms= list(split= "information"))

### Definir workflow----
w_rf <- workflow() %>% 
  add_recipe(datos_procesados) %>% 
  add_model(mod_rf)

### Grilla de hiperparámetros----
hp_rf <- grid_regular(mtry(range = c(5, 120)),
                      trees(range = c(10, 100)),
                      levels= c(3, 3))

### Búsqueda de hiperparámetros----
grilla_rf <- tune_grid(object = w_rf,
                       resamples = crossv,
                       metrics = metric_set(roc_auc),
                       control = control_resamples(save_pred = TRUE),
                       grid= hp_rf)

### Definir mejor modelo----
rf_best <- finalize_workflow(x= w_rf,
                             parameters= select_best(grilla_rf, metric= "roc_auc")) %>% 
  fit(lluvia_train_std)

### Print modelo
rf_best


# 3. Comparación Modelos--------------------------------------------------------

##3.1 Realizar predicciones ---------------------------------------

## Crear un dataframe con las predicciones de los modelos recién creados bajo el criterio auc
pred_svm <- predict(svm_best, new_data = lluvia_test_std, type = "prob")
pred_arbol <- predict(dt_best, new_data = lluvia_test_std, type = "prob")
pred_rf <- predict(rf_best, new_data = lluvia_test_std, type = "prob")

roc_auc(data  = pred_svm,
        truth = lluvia_test_std$LluviaMan_Yes,
        estimate  = .pred_1,
        estimator = 'binary',
        event_level = "second")

roc_auc(data  = pred_arbol,
        truth = lluvia_test_std$LluviaMan_Yes,
        estimate  = .pred_1,
        estimator = 'binary',
        event_level = "second")

roc_auc(data  = pred_rf,
        truth = lluvia_test_std$LluviaMan_Yes,
        estimate  = .pred_1,
        estimator = 'binary',
        event_level = "second")


## 3.2 Resultados predicciones -------------------------------------------

### ROC_AUC Support Vector Machine: 0.900
### ROC_AUC Árbol de Decisión: 0.849
### ROC_AUC Random Forest: 0.890

### De acuerdo a los datos proporcionados, el mejor modelo en esta ocasión sería el de Support Vector Machine


# 4. Análisis de Componentes Principales ----------------------------------

## Cargar dataframe----
data <- read.delim("data/datos_clustering.csv", sep = ",")
## Eliminar la variable "y"----
data_comps <- data %>% select(-y)
## Generar el PCA a partir del dataframe----
PCA <- prcomp(data_comps)
## Obtener varianza acumulada a partir del PCA----
var_acum <- data_frame(componentes = 1:10,
                       varianza_acumulada = get_eigenvalue(PCA)$cumulative.variance.percent)
## Plotear la varianza acumulada----
var_acum %>% 
  ggplot(., aes(x= componentes,
                y = varianza_acumulada))+
  geom_line(col = "purple", lwd= 1)+
  theme_bw()+
  geom_hline(yintercept = 70)

## Los datos indican que a partir de 2 componentes se puede explicar el 70% de variabilidad de los datos

## Crear dataset con 2 componentes----
data_comps_70 <- data_comps %>%  select(V1, V2)


# 5. Determinar # Clusters Ideales ---------------------------------------------------

## Estandarizar proyecciones----
proyeccion <- data_comps_70[1:1000,] %>% scale()

## Plot Elbow----
fviz_nbclust(proyeccion, kmeans, method = "wss") +
  labs(subtitle ="Gráfico de Elbow")

## Plot Silhouette----
fviz_nbclust(proyeccion, kmeans, method = "silhouette")+
  labs(subtitle ="Gráfico de Silhouette")

## Conclusiones clustering----

## Plot Elbow: El número de clusters ideales es 4
## Plot Silhouette: El número de clusters ideales es 4


# 6. K-Means Clustering ------------------------------------------------------

## Definir tipo de modelo----
mod_kmeans <- kmeans(x = proyeccion, centers = 4)

## Plots comparación----
datos_graf <- data.frame(PCA1= proyeccion[,1],
                         PCA2= proyeccion[,2],
                         Cluster = mod_kmeans$cluster,
                         y = data$y)

g1 <- ggplot(data = datos_graf, aes(x = PCA1, 
                                    y = PCA2, 
                                    color = factor(Cluster))) +
  geom_point() +
  theme_bw() 

g2 <- ggplot(data = datos_graf, aes(x = PCA1, 
                                    y = PCA2, 
                                    color = factor(y))) +
  geom_point() +
  theme_bw() 

gridExtra::grid.arrange(g1,g2)


## Conclusiones ------------------------------------------------------------

## Luego de observar los gráficos, se puede inferir que la metodología es la apropiada ya que la segmentación de PCA es similar a la original, 
## teniendo esta última un factor adicional. Sin embargo son muy pocos los puntos que cambian de cluster entre la original y la generada con PCA.
