# Run of show: 180 minutos, Colab edición

Todas las personas participantes trabajan en Colab. El instructor comparte una
pantalla de Colab con un runtime de CPU al inicio y uno con GPU a partir del bloque
5. Conviene mantener abiertos `solutions/` y este archivo en otra ventana. Si la
sesión se retrasa, se omiten primero la demostración de perfilado (bloque 2, 5
minutos) y luego el experimento con float32 (bloque 6, 5 minutos). Se conservan las
comprobaciones de corrección, la explicación de tiempos y transferencias en GPU, el
proyecto final y la discusión de cierre. La configuración de GPU termina al cumplir
los diez minutos; a las 01:30, quienes no tengan GPU continúan con el fallback.

Grabación del demo de GPU: *(añade enlace después de la puerta 5 en validation.md)*.

## 00:00 Bienvenida y preparación (10 minutos)

- 00:00 Resultados en una diapositiva (los cuatro en el README). "¡Éxito por producir resultados correctos y explicar tus mediciones. No hay un objetivo de aceleración."
- 00:02 Revisión de preparación: todos abren `01_measure_and_multicore` y ejecutan el celda de configuración. Levantan la mano cuando imprime el diccionario de tiempo. Mientras tanto:
  - **Triage.** La configuración falla por pip: pídeles que vuelvan a ejecutar la celda una vez (red de red transient); aún fallando, los pares con un compañero para este bloque y vuelven a intentar en el descanso. La celda se detiene en "instalando": Runtime > Reiniciar sesión, volver a ejecutar. Nada corre: Runtime > Desconectar y borrar el entorno, volver a abrir la enlace. No depurar individuos pasado 00:08.
  - Estudiantes que no realizaron la preparación: señalarles a la una página recapitulativa (`prep/primer.md` sección 1 y la celda "¿Qué recordar" al final de `00_stencil_practice`) y el stencil completado en `solutions/`. No repite la enseñanza de la corteza.
- 00:06 Recapitulación del stencil en la diapositiva: borde caliente, promedio de cuatro vecinos, bordes nunca cambian, el bucle de Python pura es la referencia.
- 00:08 Ejecutar juntos el apartado 1.1. Punto de control 1: la afirmación pasó.

## 00:10 Medición (25 minutos), secciones 1.2 a 1.4 del notebook 01

- 00:10 Pregunta: "¿Por qué una `time.time()` no es una medición?" Colectar tres respuestas (llamada inicial, otros procesos, resolución). Ejecutar `best_of`.
- 00:14 `%timeit`: señalar las palabras "promedio +- desviación estándar" en la salida. Error común: que `%timeit` reporta el mejor tiempo. Reporta el promedio de las ejecuciones; `.best` está disponible si se pide.
- 00:17 Predicción, ejecución, explicación: ratio de 128 vs 256. Respuesta esperada 4x, y la explicación es "tiempo proporcional a los celdas, overhead del intérprete por celda".
- Colab CPU tiempos de ejecución para Python pura son 50 a 150 ms a 128.
- 00:21 Demo de perfilado (CUT 1). Mostrar `%lprun` una vez; decir qué se busca en el código real. Silencio si llega tarde.
- 00:26 NumPy: vistas versus intermediarios. Preguntar: "¿Cuántas matrices se asignan con esta línea?" Respuesta: cuatro (tres sumas y el producto); las cortesías asignan nada. Ejecutar la tabla de referencia. Punto de control 2: todos tienen tres filas de NumPy.
- 00:33 Preguntar por la sección 1.5: "¿Es NumPy limitado por aritmética o por tráfico de memoria?" No responder aún.

## 00:35 Numba, hilos, Amdahl (35 minutos), secciones 1.5 a 1.7

- 00:35 Numba serial. Emphasizar la línea de "compilar en la primera llamada" en la salida. Error común: "Numba es más rápido porque es paralelo". No es paralelo aquí; es más rápido porque detiene la asignación de intermediarios.
- 00:42 `prange` y la lista de hilos. Mostrar `MAX_THREADS` en el lienzo compartido: una CPU Colab runtime usualmente reporta 2. Explicar por qué el cuaderno pregunta al entorno en lugar de asumir.
- 00:46 Predicción, ejecución, explicación: "¿Dos hilos reducirán el tiempo en la mitad?" Resultado típico de Colab T4: entre 1.0x y 1.6x; a veces más lento. Explicaciones para sacar: núcleo físico compartido, velocidad de memoria, arranque de hilos en un pequeño grid.
- 00:52 Grabación de la recorrida de hilos si el entorno tiene un núcleo. Decir a voz alta de qué máquina la grabación proviene (el cuaderno imprime).
- 00:55 Gráfico de Amdahl. Ejercicio: elige la curva más cercana a la medición y dice qué "fracaso serial" implica; luego explica por qué es un modelo y la velocidad de memoria es la causa más probable para este stencil.
- 01:02 Límite de conciencia del entorno: `set_num_threads(min(wanted, cpu_count))`. Sobrecarga silenciosa.
- 01:05 Punto de control 3: guardar el CSV. Preguntar a dos estudiantes para leer una fila de su tabla y su ratio Numba-NumPy. Diferentes números, misma forma: es el punto.

## 01:10 Descanso (10 minutos)

Decir a todos que guarden (Ctrl+S). Los estudiantes que quieren probar el GPU pueden hacerlo temprano, pero el bloque comienza a las 13:20 en cualquier caso.

## 01:20 Configuración de GPU (10 minutos, límite de tiempo), celda de configuración del notebook 02

- 01:20 Runtime > Cambiar tipo de entorno > GPU. Ejecutar la celda de configuración. Imprime `MODE: GPU` o `MODE: CPU fallback`.
- Triage: "no GPU disponible" de Colab: fallback, inmediatamente. Error de importación de CuPy en un entorno de GPU: volver a ejecutar una vez; aún fallando, fallback. Reiniciar los bucles: fallback. No gastar el tiempo del entorno en una máquina.
- 01:28 Mostrar el banner de fallback en pantalla para que los estudiantes sepan qué verán: filas de CPU vivas, filas de GPU grabadas con etiqueta de hardware, y las mismas preguntas.
- 01:30 Pasar a lo que esté ocurriendo en el entorno.

## 01:30 CuPy, sincronización, recorrida, transferencias, precisión (40 minutos), secciones 2.1 a 2.6

- 01:30 Sección 2.1. El patrón `xp`. Punto de control 1: los mismos números en una memoria diferente.
- 01:35 Sección 2.2. Ejecutar la celda de no-sinc versus sinc. Error común a surfear: "el GPU es tan rápido que el tiempo es cero". Preguntar qué contenía el reloj.
- 01:41 Sección 2.3 predecir: "¿A qué n el GPU superará al CPU en este entorno?" Ejecutar. Preguntas de explicación 1 y 2. No establecer un tamaño de crossover; preguntar al cuarto por su tamaño, luego por el de la tabla de fallback, y señalar que difieren.
- 01:52 Sección 2.4 transferencias. Predicción: subir más caro o más barato que un paso? Respuesta típica de Colab T4: un subir de 32 MB cuesta varios pasos de corteza. Mostrar los dos escopetas lado a lado. Regla: señalar qué escopeta se cita.
- 02:01 Sección 2.5 float32 (CUT 2). Puntar el chequeo con una tolerancia más laxa, y que el ratio f64/f32 es una propiedad del recurso al que estás ligado, no de "el GPU".
- 02:06 Sección 2.6 brevemente, y Punto de control 2: guardar el CSV. Pregunta de salida: tres cosas para preguntar sobre "40x más rápido" (sincronización incluida, mismo trabajo y tipo de datos).

## 02:10 Más allá de una máquina (20 minutos), notebook 03

- 02:10 Una diapositiva: máquinas separadas, memorias separadas. Los dos modelos de programación.
- 02:13 Sección 3.2 ejercicio antes de ejecutar: filas por borde por paso, y bytes. Respuesta: dos filas por borde interno, `2*(P-1)*n*8` bytes en total, `2n*8` por hilo en el medio. Ejecutar; el resultado reconstruido es `array_equal`, no solo cercano.
- 02:19 Sección 3.3 modelo y la pregunta de latencia. Respuesta esperada: a n=256, P=128 un hilo computa 512 celdas (cerca de 0.5 us) y espera por dos mensajes (cerca de 20 us de latencia): principalmente esperando. A n=16384 computa 2 millones de celdas (2 ms) por dos mensajes: principalmente computando.
- 02:24 Secciones 3.4 y 3.5 como lectura con el código en pantalla: unir cada línea de `Sendrecv` con una línea de `exchange_halos`; luego la recorrida sin halos. Establecer claramente que este cuaderno no hace ninguna afirmación de rendimiento multi-nodo.
- 02:28 Punto de control: una frase por cada carga de trabajo capstone, que fila de la tabla.

## 02:30 Capstone (20 minutos), notebook 04

- 02:30 Dos cargas de trabajo, tres entregables cada. Los estudiantes editan `CHOICE_A` y `CHOICE_B`, ejecutan, y escriben la celda de recomendación. Circulan.
- Resultados esperados y razonamiento: `solutions/capstone_solution.md`. La sorpresa enseñable es la carga de trabajo A, donde `numba_par` a menudo es más lento que `numba`.
- 02:45 Dos voluntarios leen su recomendación. Preguntar a cada uno: "¿Qué número en tu tabla es la base de esta oración?"
- 02:48 Guardar el CSV y el cuaderno.

## 02:50 Revisión (10 minutos)

- Pregunta de salida: "¿Cuándo más hardware no ayudaría?" Colectar: carga de trabajo pequeña (lancamiento/hilo overhead), transferencias dentro de un bucle, kernel limitado por memoria, fracaso serial, descomposición comunicacional.
- La diapositiva de la tabla de decisiones. A dónde ir después: las extensiones, la guía de lectura, el enlace a las diapositivas.
