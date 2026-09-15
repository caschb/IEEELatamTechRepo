# Self-check (5 to 10 minutes)

Responde a cada pregunta antes de abrir su explicación. Tener una respuesta incorrecta es
perfectamente normal; lee la explicación y continúa. Estas preguntas cubren lo que asumen los
primeros diez minutos de vida.

## 1. Cortado

`u` es un array de 6x6. ¿Cuál expresión tiene la misma forma que `u[1:-1, 1:-1]`
y selecciona, para cada celda interior, la celda a su **izquierda**?

(a) `u[1:-1, 2:]`  (b) `u[1:-1, :-2]`  (c) `u[:-2, 1:-1]`  (d) `u[:, :-2]`

<details><summary>Respuesta</summary>

**(b).** Mantén las filas la misma (`1:-1`) y mueve las columnas una unidad a la izquierda:
`:-2` selecciona columnas 0 a 3, que son los vecinos izquierdos de las columnas 1 a 4.
(a) es el vecino derecho, (c) el uno encima, (d) tiene 6 filas y por tanto no puede ser
agregado a la 4x4 interior.
</details>

## 2. Preservación de la frontera

En el stencil, `unew[1:-1, 1:-1] = 0.25 * (...)` actualiza solo el interior.
Después de 100 pasos, ¿qué valor tiene la fila superior `u[0, :]`?

(a) Ha enfriado hacia el promedio  (b) Todavía 100 en todas partes  (c) 25  (d) Indefinido

<details><summary>Respuesta</summary>

**(b).** La fila superior nunca se escribe, así que permanece en 100. Es la condición
de frontera, la "borde caliente" que impulsa toda la simulación. Lo mismo es cierto
de los otros tres bordes, que permanecen en 0. Cualquier implementación más rápida
debe preservar esto, y los cuadernos lo comproban explícitamente.
</details>

## 3. Aceleración

Versión A tarda 8.0 s. Versión B tarda 0.5 s en la misma máquina. Un colega
ejecuta versión B en un portátil más rápido en 0.2 s. ¿Cuál es la aceleración de B sobre A?

(a) 40x  (b) 16x  (c) 2.5x  (d) No se puede decir sin más información

<details><summary>Respuesta</summary>

**(b).** 8.0 / 0.5 = 16, medida en la misma máquina. El número de portátil es un
experimento diferente: hardware diferente, así que no puede compararse con A. En
la oficina de trabajo cada comparación se hace dentro de un solo runtime.
</details>

## 4. Memoria separada del dispositivo

`x` es un array de NumPy en RAM. ¿Qué declaración sobre una computación en GPU sobre él
es verdadera?

(a) El GPU lee `x` directamente de RAM, así que no es necesario hacer una copia
(b) `x` debe ser copiado a la memoria del GPU primero; el resultado debe ser copiado de vuelta para ser usado en NumPy
(c) Las copias son tan rápidas que nunca importan
(d) Las arrays de CuPy y arrays de NumPy comparten memoria

<details><summary>Respuesta</summary>

**(b).** El GPU tiene su propia memoria. `cp.asarray(x)` copia hacia arriba; `.get()`
copia hacia abajo. Las copias pasan a través de un enlace que es mucho más lento que
ambas memorias, así que un bucle que copia cada iteración puede ser más lento que no usar
el GPU en absoluto.
</details>

## 5. Ámbito de medición de tiempo

¿Quién quiere medir 20 pasos de stencil en el GPU? ¿Cuál procedimiento es correcto?

(a) Empieza el cronómetro, ejecuta 20 pasos, detiene el cronómetro
(b) Ejecuta 20 pasos una vez para calentar; luego empieza el cronómetro, ejecuta 20 pasos, detiene el cronómetro
(c) Ejecuta 20 pasos una vez para calentar; sincroniza; empieza el cronómetro, ejecuta 20 pasos, sincroniza, detiene el cronómetro; repite varias veces y reporta el mínimo o mediana
(d) Tiempo un paso y multiplica por 20

<details><summary>Respuesta</summary>

**(c).** Calentar elimina la compilación y la asignación de la medición.
Sincronizar antes de detener el cronómetro es esencial en un GPU, porque las llamadas
devuelven antes de que el trabajo esté hecho; sin él, tiempo la *lanzamiento*.
Las repeticiones muestran la dispersión. (d) omite el hecho de que el primer paso no
es representativo y que el overhead por paso puede dominar a pequeños tamaños.
</details>

## Una más, para la discusión al final de la oficina de trabajo

¿Cuándo comprar más hardware **no** hará que un programa sea más rápido? Piensa en dos
razones. (¡Tendrás que enfrentarte a al menos tres durante la sesión en vivo.)
