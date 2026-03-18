#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importar las librerias de uso común
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats as st


# ### Abrir archivo y analizar su información general
# - **Importar** archivo
# - Aplicar función **.info()**
# - Analizar los **valores nulos**

# In[2]:


#Importar archivo
games = pd.read_csv('games.csv')


# In[3]:


print(games.info())


# In[4]:


#Valores nulos por columna
print(games.isna().sum())


# In[5]:


print(games)


# ### Preparación de Datos
# - Nombres de columnas estandarizados (minuscula y con snake_case)
# - Convertir datos en formatos requeridos
# - Describir y justificar las columnas en las que se cambio el formato de dato
# - Justificación y explicación de los valores ausentes, y del manejo de los valores TBD.
# - Calculo de las ventas totales por juego en la columna 'total_sales'

# In[6]:


#Estandarizar nombres de columnas
col = []
for name in games.columns:
    col.append(name.lower().strip())

games.columns = col


# In[7]:


print(games.columns)


# In[8]:


#Formatos de datos requeridos
games['platform'] = games['platform'].astype('category')
games['rating'].fillna('uknown', inplace=True)
games['rating'] = games['rating'].astype('category')
games['year_of_release' ] = games['year_of_release'].astype(float)


# Se modificaron las columnas 'platform' y 'rating' a tipo 'category' con el objetivo de volver más ligero el dataset. y la columna 'year_of_release' a float ya que es de tipo numerico.

# In[9]:


#Manejar valores ausentes 
games['name'].fillna('uknown', inplace=True)
games['genre'].fillna('uknown', inplace=True)
games = games.replace('tbd', np.nan)
games['user_score'] = games['user_score'].astype(float)


# Los valores ausentes en las columnas 'name' y 'genre' fueron reemplazados por 'uknown' para así poder agruparlos y analizarlos.
# Los str 'tbd' en 'user_score' fueron reemplazados por valores nulos ya que se encuentrar en filas numericas y ya con este cambio nos permite transformar la columna a valores int y se justifica ya que 'tbd' es un valor aun no definido y por lo tanto nulo. Posteriormente fue cambiado a formato float.

# In[10]:


#Calculo de ventas totales por fila en la columna 'total_sales'
games['total_sales'] = games['na_sales'] + games['eu_sales'] + games['jp_sales'] + games['other_sales']
print(games[['name', 'total_sales']].sample(4))


# ### Analisis de datos
# - **Cantidad** de juegos lanzados por año
# - Variación de **ventas por plataforma**
# - **Distribución** basada en los datos de cada año de las **plataformas con más ventas totales**
# - Analisis de las plataformas que **fueron populares**, pero **ya no tienen ventas**
# - **Tiempo** que toman las nuevas plataformas en **surgir** y las antiguas en **desaparecer**.
# - **Determianción del periodo** que se tomaran datos (de acuerdo a los puntos anteriores) para el **modelo de 2017** (Solo se trabajara con los datos reelevantes, se **descartan los de años anteriores**)
# - Plataformas **lideres en ventas**, cuales crecen y cuales se reducen
# - Selección de **plataformas potencialmente rentables**
# - Diagrama de caja para **ventas globales por plataforma** (*¿Son significativas las diferencias en las ventas? ¿Qué sucede con las ventas promedio en varias plataformas? Descripción de hallazgos*.)
# - **Impacto de reseñas profesionales y de usuarios** en las ventas de una plataforma popular seleccionada (con grafico de dispersión y correlación entre reseñas y ventas)
# - Comparación de las **ventas de los mismos juegos por pltaforma**
# - **Distribución general de juegos por género** (Analisis de generos más y menos rentables)

# In[11]:


#Cantidad de juegos lanzados por año
yearly_launches = games.groupby('year_of_release').agg({'na_sales': 'sum',
                                                        'eu_sales': 'sum',
                                                        'jp_sales': 'sum',
                                                        'other_sales': 'sum',
                                                        'name': 'count',
                                                        'total_sales': 'sum'
                                                        })

#Generamos una grafica de barras
yearly_launches.name.plot(kind='bar',
                         color='lightgreen',
                         width=0.9,
                         figsize=(13,4)
                         )

plt.ylabel('Juegos lanzados')
plt.xlabel('Año')
plt.grid(True)
plt.xticks(rotation=60) 
plt.show()


# - De 2002 a 2010 fue cuando más juegos se lanzaron por año, superando cada año los 600 juegos.

# In[12]:


#Variación de ventas por plataforma

# Agrupamos por plataforma y sumamos las ventas
sales_by_platform = games.groupby('platform')['total_sales'].sum().sort_values(ascending=False)

print(sales_by_platform)


# In[13]:


#Distribución basada en los datos de cada año de las plataformas con más ventas totales
# Graficamos el resultado
sales_by_platform[sales_by_platform>400].plot(kind='bar',
                                              figsize=(13, 4),
                                              color='lightgreen')
plt.title('Ventas totales por plataforma')
plt.ylabel('Ventas (millones)')
plt.xticks(rotation=0) 
plt.grid(True)

plt.show()


# - Como podemos apreciar las 6 plataformas mas importantes son **PS2, X360, PS3, Wii, DS y PS**, ya que todas superaron los 700 millones.

# In[14]:


#Analisis de las plataformas que fueron populares, pero ya no tienen ventas
# Agrupamos por año y sumamos las ventas
platform_sales_by_year = games.groupby(['platform', 'year_of_release'])['total_sales'].sum()

fig, ax = plt.subplots(figsize=(13,4))

# Filtramos las plataformas con ventas > 400 o algun año con más de 15 millones
more_tan_15_millions = games.groupby('platform')['total_sales'].max().sort_values(ascending=False) >15
platforms = sales_by_platform[sales_by_platform > 400].index
platforms = (more_tan_15_millions[more_tan_15_millions].index[:].append(platforms)).unique()

# Mapeo exacto para mantener consistencia en tus gráficas
platform_colors = {
    'Wii':  '#8fe2ff', # Celeste/Blanco interfaz
    'NES':  '#d3d3d3', # Gris ceniza original
    'GB':   '#8b9010', # Verde oliva de la pantalla original
    'DS':   '#6d6e71', # Gris metálico
    'X360': '#5cb85c', # Verde Xbox
    'PS3':  '#000000', # Negro Piano
    'PS2':  '#003791', # Azul PlayStation
    'SNES': '#808285', # Gris oscuro de los botones
    'GBA':  '#4b0082', # Morado índigo (modelo icónico)
    'PS':   '#dfdfdf'  # Gris claro (Classic)
}

# Para usarlo en tu bucle actual:
#platforms = ['Wii', 'NES', 'GB', 'DS', 'X360', 'PS3', 'PS2', 'SNES', 'GBA', 'PS']
colors = [platform_colors[p] for p in platforms]

# Usamos zip para emparejar cada plataforma con un color
for platform, color in zip(platforms, colors):
    if platform != 'DS':
        platform_sales_by_year.loc[platform][platform_sales_by_year.loc[platform] != 0].plot(color=color)
    else: #Se detecto un error en las ventas de DS en el 1985, ya que marca 0.02 millones, pero en realidad salio hasta 2004, para esto eliminamos los valores anteriores a su lanzamiento
        platform_sales_by_year.loc[platform][platform_sales_by_year.loc[platform] != 0].iloc[1:].plot(color=color)

plt.title('Ventas por año de cada plataforma que tuvo ventas de más de 400 millones en total o algun año con ventas de más de 15 millones')
plt.legend(platforms)
plt.grid(True)
plt.show()


# - Aqui podemos observar 10 plataformas populares que actualmente ya casi no tienen ventas, Todas parecen tener un comportamiento similar en sus ventas, a continuación analizaremos el tiempo que toman en surgir y desaparecer.

# In[15]:


#Analisis del tiempo que toman en surgir y desaparecer las plataformas
top_10 = platform_sales_by_year[platform_sales_by_year != 0]

data = []
for platform in platforms:

    if platform != 'DS':
        years = top_10.loc[platform].index
    else:
        years = top_10.loc[platform].iloc[1:].index
    first, last, top = years.min(), years.max(), top_10.loc[platform].max()
    top = top_10[top_10 == top].reset_index()['year_of_release'][0] - first
    end = last - first + 1 #Se suma un año, ya que last es el ultimo año que se registraron ventas y nosotros buscamos el año siguiente a ese, que fue en el que ya no se vendio nada
    data.append({'platform': platform, 'mejor_año': top, 'dejo_de_vender': end})

df = pd.DataFrame(data).set_index('platform')


# In[16]:


#Grafica de caja para el mejor año por platafroma y los años que pasan hasta que deja de vender
df.boxplot()
plt.title('Surgimiento y decadencia de plataformas')
plt.xticks([1, 2], ['Mejor año', 'Se dejó de vender'])
plt.ylabel('Años del Ciclo de Vida')
plt.show()

#Mostramos estadistica descriptiva
print('\nPromedio')
print(df.mean(), '\n')

print('Desviación estandar')
print(round(np.sqrt(df.var()), 2))


# - Gracias al analisis estadistico podemos concluir que los juegos de las plataformas exitosas tuvieron su mejor año en promedio depues de 3.4 años del lanzamiento de la misma, con una desviación estandar de 1.26 y en promedio dejaron de vender despúes de 11 años de su lanzamiento, con una desviación estandar de 1.63.
# - Además gracias a la grafica de caja podemos concluir que el mejor año de una plataforma exitosa va del año 2 al año 5, aproximadamente, mientras va de 8 a 14 años el tiempo que pasa para que esa plataforma deje de vender juegos.

# ### Determinación del periodo que se tomara de datos para construir modelo del 2017 (de acuerdo a respuestas anteriros)
# 
# - Sería conveniente tomar los datos que abarquen lo suficiente para ver una plataforma surgir y desaparecer, para lo cual tomaremos el máximo de lo que le toma a una plataforma para desaparecer, la cual es de 14 años.

# In[17]:


#Ventas promedio de los ultimos 14 años
games.groupby('year_of_release')['total_sales'].mean().loc[2016-14:].plot()


# In[18]:


#DF de ventas de juegos lanzados del 2002 en adelante
df_2002_en_adelante = games[(games['year_of_release'] >=2002) & (games['total_sales'] > 0)]


# #### Plataformas lideres

# In[19]:


#Plataformas lideres en ventas, cuales crecen y cuales se reducen
grouped_2_to_16 = df_2002_en_adelante.groupby(['platform', 'year_of_release'])[['total_sales']].sum()

# 'unstack' mueve el nivel del índice 'platform' a las columnas
df_final = grouped_2_to_16['total_sales'].unstack(level='platform').fillna(0)

# Solo los que más vendieron los ultimos 3 años 
df_final = df_final[df_final.index >= 2014]
df_final = df_final.loc[:, (df_final != 0).any(axis=0)].sum()
leaders, loosers = df_final.sort_values().index[-3:], df_final.sort_values().index[:3]

#Imprimir plataformas lider y su promedio de venta los ultimos 3 años
print('\n', 'Plataformas lider')
print(df_final.loc[leaders])

#Imprimir plataformas que se reducen y su promedio de venta los ultimos 3 años
print('\n', 'Plataformas')
print(df_final.loc[loosers])


# In[20]:


#Exztraemos los años que tienen en el mercado las plataformas lideres
for leader in leaders:
    print('\n', leader, grouped_2_to_16[grouped_2_to_16 > 0].loc[leader].count())


# Las plataformas potencialmente rentables serian **3DS**, **XOne**, **PS4** de acuerdo a las ventas de lo ultimos 3 años. Pero 3DS ya tiene 6 años en el mercado, signofica que ya pasaron sus mejores años, por ello nos quedaremos con **XOne** y **PS4** como plataformas con gran potencial.

# In[21]:


# Plataformas con ventas de 2002 en adelante
sales = df_2002_en_adelante.groupby('platform')['total_sales'].sum()
sales = sales[sales != 0]

# Eliminar plataformas sin ventas
df = grouped_2_to_16.reset_index()
df['platform'] = df['platform'].astype(str)
df = df.query('platform in @sales.index')


# #### Diagrama de caja para las ventas globales de todos los juegos, desglosados por plataforma.

# In[22]:


df.boxplot(column='total_sales', by='platform', figsize=(13,4), rot=45)

plt.title('Distribución anual de ventas por plataforma del 2002 al 2016')
plt.ylabel('Ventas en Millones ($)')
plt.xlabel('Plataforma')
plt.suptitle('')
plt.show()


# In[25]:


mean_ = df[df['total_sales'] > 0].groupby('platform')[['total_sales']].mean()
std = np.sqrt(df[df['total_sales'] > 0].groupby('platform')[['total_sales']].var())

print('Media\n', mean_)

print('\nDesviación\n', std)


# In[26]:


fig, ax = plt.subplots(figsize = [10, 4])
mean_.plot(ax=ax,
          kind='bar',
          width=0.9,
          alpha=0.8)

std.plot(ax=ax,
         kind='bar',
         width=0.9,
         color='red',
        alpha=0.8)

plt.title('Medía y desviación estandar de las ventas anuales desde el 2002')
plt.legend(['Media', 'Desviación'])
plt.show()


# - La diferencia en las ventas es muy notoría en los diagramas de caja, podemos ver como resaltan **DS, PS2, PS3, Wii, X360** con gran diferencia.
# - La media de las ventas de **DS, PS2, PS3, PS4, Wii, X360** son notoriamente más altas que el resto de las plataformas pero tienen una desviación bastante alta, casi tanta como el resto de las plataformas. 

# #### Impacto de las reseñas profesionales y de los usuarios en PS4 y XOne

# In[27]:


#Impacto de reseñas profesionales y de usuarios en las ventas de una plataforma popular seleccionada 
df_ = df_2002_en_adelante.copy()
df_.loc[:, 'user_score'] = df_['user_score'] * 10

def score_scatter_plot(df, platform):
    ### Grafico de dispersion de impacto de las calificaciones en las ventas 
    fig, ax = plt.subplots(figsize=[4,4])

    df = df[df.loc[:,'platform'] == platform]
    df.plot(kind='scatter',
              x='critic_score',
              y='total_sales',
              alpha=0.4,
              ax=ax)
    
    df.plot(kind='scatter',
              x='user_score',
              y='total_sales',
              alpha=0.4,
              ax=ax,
              color='red')
    plt.legend(['Criticos', 'Usuarios'])
    plt.title(f'Dispersión de ventas por calificación de cricitos y usuarios para {platform}')
    plt.xlabel('Calificación')
    plt.ylabel('Ventas por juego en millones')
    plt.show()

    print('Correlación de ventas totales con:')
    print(f"Criticos {round(df['critic_score'].corr(df['total_sales']), 2)}")
    print(f"Usuarios {round(df['user_score'].corr(df['total_sales']), 2)}, \n")

score_scatter_plot(df_, 'PS4')

score_scatter_plot(df_, 'XOne')


# - La correlación no es suficientemente significativa en ninguno de los casos en ambas plataformas, ya que no alcanzan el 0.5, esto se confirma con las graficas de dispersión.

# #### Comparación de las ventas de los mismos juegos por pltaforma

# In[28]:


# Juegos populares
game_platform = df_[df_ !=0].groupby('name')['total_sales'].sum().sort_values()
print(game_platform[game_platform > 25],'\n')

print(game_platform[game_platform > 25].index)


# - Seleccionamos los juegos más populares

# In[29]:


# Contamos en cuántas plataformas aparece cada nombre de juego
multiplatform_games = games.groupby('name')['platform'].nunique()

# Nos quedamos solo con los nombres que aparecen en 2 o más
multi_names = multiplatform_games[multiplatform_games > 1].index

# Filtramos el DataFrame original para tener solo esos juegos
df_multi = games[games['name'].isin(multi_names)]


# - Ahora con los juegos multi-plataforma más populares hacemos una tabla de pivote

# In[30]:


# Creamos una tabla pivote
comparison_table = df_multi.pivot_table(index='name', 
                                        columns='platform', 
                                        values='total_sales', 
                                        aggfunc='sum')

# Usamos los más populares multi-plataforma
names = multi_names.intersection(game_platform[game_platform > 25].index)
#print(comparison_table.loc[names].T.loc[mean_.index])


# In[31]:


comparison_table.loc[names].loc[:,mean_.index].plot(kind='bar',
                                                   figsize=(13,6),
                                                   rot=70,
                                                   width=0.9,
                                                   alpha=0.7)
plt.title('Ventas por juego por plataforma')
plt.xlabel('Juego')
plt.ylabel('Ventas en millones')
plt.show()


# - Como podemos notar, tanto PS3 como X360, tienen un notorio dominio en 5 de los 6 juegos seleccionados, mientras en uno de los juegos estan depues de otros dos, los cuales son las versiones más recientes de estas dos plataformas.

# In[32]:


#Distribución general de juegos por género (Analisis de generos más y menos rentables)
by_genre = df_.groupby('genre')[['total_sales']].mean().sort_values('total_sales',ascending=False)
print(by_genre, '\n')

by_genre.plot(kind='bar',
             title='Ventas por genero',
             alpha=0.8,
             xlabel='Genero',
             ylabel='Ventas en millones',
             grid=True)

plt.legend('')
plt.show()


# - Notoriamente los 2 generos más rentables son **Shooter y Platform** mientras los generos menos rentables serian **Strategy y Adventure**, con menos de un tercio de las ventas de los anteriormente mencionados *Shooter y Platform*.

# ### Perfil de usuario por region
# 
# Para cada región (NA, UE, JP) se determina:
# 
# - Las cinco plataformas principales. Variaciones en sus cuotas de mercado de una región a otra.
# - Los cinco géneros principales.
# - Impacto de las clasificaciones de ESRB en las ventas en regiones individuales.

# In[33]:


# Funciones para perfil 
regiones = ['na_sales', 'eu_sales', 'jp_sales']

def top_5_region(df, regional, search):
    df = df.groupby(search)[[regional]].sum().sort_values(regional, ascending=False).head()
    df['porcentaje'] = round(df / df.sum(), 2)
    return df


# In[34]:


# 5 principales plataformas por region
for reg in regiones:
    print(top_5_region(df_, reg, 'platform'), '\n')


# - Es notorio que cada region tiene su plataforma favorita, X360 para NA, PS3 en EU y DS en JP.

# In[35]:


# 5 principales generos
for reg in regiones:
    print(top_5_region(df_, reg, 'genre'), '\n')


# - En ninguna de las regiones observadas coincide el principal genero con el total global y aun así cada genero principal por region pose más del 30% de las ventas por region.

# In[36]:


# Impacto de las clasificaciones ESRB en las ventas regionales
for reg in regiones:
    print(top_5_region(df_, reg, 'rating'), '\n')


# In[37]:


print('Distribución global de clasificaciones\n')
global_esrb = df_.groupby('rating')[['total_sales']].sum().sort_values('total_sales', ascending=False).head()
print(global_esrb / global_esrb.sum())


# - La clasificación **E** pose **35%, 33% y 24%** de las ventas regionales, acercandose bastante al **32.6% global**, además en todas las regiones coincide el Top 5, en diferente orden.

# ### Pruebas de hipótesis

# In[38]:


# Funcion para prueba de hipotesis de igualdad de 2 data set
def user_score_hipotesis(df, searches, in_, alpha=0.05):
    # Generar los 2 data set separados
    xone_ratings = df[df[in_] == searches[0]]['user_score'].dropna()
    pc_ratings = df[df[in_] == searches[1]]['user_score'].dropna()
    
    # Comparamos las varianzas de las calificaciones
    levene_res = st.levene(xone_ratings, pc_ratings)
    print('p-value de Levene:', levene_res.pvalue)
    
    if levene_res.pvalue < 0.05: equal_var = False
    else: equal_var = True
    
    #Prueba de hipótesis
    results = st.ttest_ind(xone_ratings, pc_ratings, equal_var=equal_var)
    
    print('p-value:', results.pvalue)
    
    if results.pvalue < alpha:
        print("\nRechazamos la hipótesis nula: Las calificaciones son diferentes.")
    else:
        print("\nNo podemos rechazar la hipótesis nula: Las calificaciones son similares.")


# #### Hipotesis 1
# - H0: Las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas.

# In[39]:


user_score_hipotesis(df_, ['XOne', 'PC'], 'platform')


# #### Formulación de hipótesis nula y alternativa.
# - La hipotesis nula siempre se hace sobre una igualdad, al estar interesados en saber si **XOne y PC tienen calificaciones iguales**, determinamos que H0 sería determinar si existe una igualdad y por lo tanto h1 sería que son diferentes. ***Hipótesis nula rechazada***.
# 
# #### Criterios utilizados para probar las hipótesis.
# - Tomamos **Alpha de 0.05** para darle un poco de flexibilidad ya que la prueba de levene nos dio un p_value >0.2 y además, hay bastantes juegos que no se repiten en ambas plataformas.
# - Usamos la prueba de **levene** para determianar si tienen una **varianza igual** y determiar correctamente el parametro en la prueba de hipótesis.
# - Se ejecuta la prueba **ttest_ind** porque es justo la prueba ideal cuando se busca comparar dos data set.

# #### Hipótesis 2
# - H0: Las calificaciones promedio de los usuarios para los géneros de Acción y Deportes son iguales.

# In[40]:


user_score_hipotesis(df_, ['Action', 'Sports'], 'genre', 0.01)


# In[41]:


#Calificación promedio del usuario para Acción y Deportes
print(df_.groupby('genre')[['user_score']].mean().loc[['Action', 'Sports']])


# #### Formulación de hipótesis nula y alternativa.
# - Al estar interesados en saber si los genero **Action y Sports tienen calificaciones iguales**, determinamos H0 sería: **las calificaciones de los usuarios Acción y Deportes son iguales** y por lo tanto h1: **las calificaciones de los usuarios Acción y Deportes son iguales**.  ***Hipótesis nula no rechazada***.
# 
# #### Criterios utilizados para probar las hipótesis.
# - Tomamos **Alpha de 0.01** ya que necesitamos una prueba más estricta para comparar calificaciones sobre los generos de los juegos. Lo cual se refuerza al notar que sus promedios de **calificación son de 70.39 y 69.04** (muy cercanos, con una *diferencia de apenas el 1.92%*). Esto nos ayuda a evitar un **Falso Positivo**.
# - Usamos la prueba de **levene** para determianar si tienen una **varianza igual** y determiar correctamente el parametro *equal_var* en la prueba de hipótesis.
# - Se ejecuta la prueba **ttest_ind** siendo la prueba ideal cuando se busca comparar dos data set.

# # Conclusión General: Análisis del Mercado de Videojuegos (2002-2016)
# 
# ### 1. Preparación y Calidad de los Datos
# El proceso de limpieza aseguró la integridad del análisis mediante la estandarización a `snake_case` y la corrección de tipos de datos. Se gestionaron los valores ausentes en **'name'** y **'genre'** como `'unknown'` para evitar sesgos en la agrupación. Un paso crítico fue la transformación de `'tbd'` en `user_score` a valores nulos, permitiendo el cálculo numérico tras normalizar la escala a un rango de 0-100 para equipararla con las críticas de expertos.
# 
# ### 2. Dinámica del Ciclo de Vida de las Plataformas
# El análisis histórico revela que el mercado alcanzó su pico de lanzamientos entre 2002 y 2010. Se identificaron las 6 plataformas líderes (**PS2, X360, PS3, Wii, DS y PS**) como los referentes históricos con ventas superiores a los 700 millones de USD.
# 
# * **Patrón de éxito:** Las plataformas tardan, en promedio, **3.4 años** en alcanzar su madurez comercial (mejor año) y tienen una vida útil total de aproximadamente **11 años**.
# * **Ventana de oportunidad:** Los datos muestran que el periodo de máximo rendimiento ocurre entre el **año 2 y el año 5** tras el lanzamiento.
# 
# ### 3. Estrategia y Pronóstico para 2017
# Para el modelo predictivo de 2017, se determinó que el periodo de relevancia debe cubrir el ciclo de vida observado (hasta 14 años para capturar la decadencia total).
# 
# * **Plataformas Objetivo:** Aunque la **3DS** muestra ventas sólidas, su antigüedad de 6 años sugiere que ha entrado en su fase de declive. Por lo tanto, **PS4 y Xbox One** se perfilan como las plataformas con mayor potencial de rentabilidad para 2017, al encontrarse aún en una etapa competitiva de su ciclo de vida.
# * **Factores de Éxito:** Los géneros **Shooter y Platform** dominan la rentabilidad global, mientras que la correlación entre las ventas y las calificaciones (críticos/usuarios) es débil (menor a 0.5), lo que indica que el éxito comercial depende más del marketing, la marca y la exclusividad que de la puntuación promedio.
# 
# ### 4. Perfilamiento Regional
# Existe una fragmentación clara en las preferencias: **X360** domina en Norteamérica, **PS3** en Europa y **DS** en Japón. A pesar de estas diferencias de hardware, la clasificación **E (Everyone)** mantiene una consistencia global, representando cerca del 32.6% de las ventas en todas las regiones.
# 
# ### 5. Verificación de Hipótesis Estadísticas
# * **Xbox One vs. PC:** Se rechazó la hipótesis nula ($p < 0.05$), confirmando que las calificaciones promedio de los usuarios **son diferentes** entre estas plataformas.
# * **Acción vs. Deportes:** Utilizando un nivel de significancia más estricto ($\alpha = 0.01$) debido a la cercanía de los promedios (solo 1.92% de diferencia), **no se pudo rechazar la hipótesis nula**. Se concluye que las calificaciones para los géneros de Acción y Deportes son estadísticamente similares.
