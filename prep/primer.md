# Primer: las cinco ideas que el taller construye sobre (15 minutos)

Lea esto antes del evento. Cada sección termina con una frase para recordar.

## 1. Arrays: forma, corte, dtype, vistas y copias

Un array de NumPy es un bloque de números de un tipo (`dtype`) con una `forma`.
`u = np.zeros((4, 6))` es 4 filas por 6 columnas de `float64`, 8 bytes cada.

El corte selecciona un rectángulo sin copiarlo. `u[1:-1, 1:-1]` es cada fila
excepto la primera y última, y cada columna excepto la primera y última: la
**interior**. Desplazando un corte por uno selecciona a los vecinos del interior:

| Corte | Selecciona |
|---|---|
| `u[:-2, 1:-1]` | la celda **arriba** de cada celda interior |
| `u[2:, 1:-1]` | la celda **abajo** de cada celda interior |
| `u[1:-1, :-2]` | la celda **izquierda** de cada celda interior |
| `u[1:-1, 2:]` | la celda **derecha** de cada celda interior |

Todos cuatro tienen la misma forma que el interior, así que pueden sumarse.

Un corte es una **vista**: comparte memoria con el original, así que escribir en él
cambie el original. La aritmética es diferente: `a + b` asigna un **nuevo
array** para el resultado, incluso si `a` y `b` son vistas. Cuatro sumas producen
cuatro arrays temporales. Eso importa para la velocidad más tarde.

`dtype` decide los bytes por número. `float64` es el tipo por defecto; `float32`
reduce el espacio de memoria y el tráfico en la mitad, a costa de precisión.

*Recuerde: los cortes son vistas gratuitas; la aritmética asigna.*

## 2. La memoria del CPU y la memoria del GPU son lugares diferentes

El GPU tiene su propia memoria, separada de la RAM del computador. Antes de que
el GPU pueda trabajar con un array, el array debe ser **copiado** al dispositivo; para
ver el resultado desde Python, debe copiarse de vuelta. La conexión entre los dos
(PCIe) es mucho más lenta que la memoria en ambos lados. CuPy, la biblioteca usada
en el taller, se ve exactamente como NumPy pero mantiene sus arrays en el GPU; las
copias son `cp.asarray(x)` (arriba) y `x.get()` (abajo).

*Recuerde: mueva la datos una vez, realiza mucho cálculo, mueva de vuelta una vez.*

## 3. La concurrencia no es paralelismo

**Concurrencia** es manejar varias cosas a la vez (tomar turnos); **paralelismo** es
hacer varias cosas al mismo tiempo (varias núcleos). Las hilas de Python son
concurrencias pero, para el código Python ordinario, no son paralelas: un bloque
de bloqueo (el GIL) permite que una hilo ejecute Python a la vez. El código compilado
(los interiores de NumPy, Numba) puede liberar ese bloqueo, así que puede ejecutarse
paralelamente en hilas. Las **procesos** separados siempre son paralelas pero no comparten
memoria, así que los datos deben copiarse a ellos.

*Recuerde: las hilas para el código compilado, los procesos para el código de Python, y
medir.*

## 4. Speedup y la ley de Amdahl

El speedup = tiempo antes / tiempo después, en la misma máquina, mismo problema. Diez
segundos a dos segundos es un speedup de 5.

Si una fracción `s` del trabajo no puede ser paralelizado, entonces con `p` núcleos:

    speedup(p) = 1 / (s + (1 - s) / p)

Ejemplo de cálculo: un programa pasa 10% de su tiempo leyendo un archivo (serial) y
90% en un bucle que paraleliza perfectamente. Con 4 núcleos:

    1 / (0.10 + 0.90 / 4) = 1 / 0.325 = 3.08

Con 16 núcleos: `1 / (0.10 + 0.9/16) = 6.4`. Con infinitos núcleos: `1 / 0.10 = 10`.
El 10% serial pone el techo del speedup a 10, independientemente de cuánto hardware
añadas. En la práctica, el techo llega antes, porque el parte paralelo también comparte
la banda ancha de memoria y paga por la coordinación.

*Recuerde: la fracción serial pone el techo; encuentrala antes de comprar núcleos.*

## 5. ¿Qué mida una medición de tiempo?

Una medición solo es significativa si puede decir qué estaba dentro del reloj:

- **Calentamiento.** La primera llamada paga por la compilación, las importaciones y los
  cachés. Tiene en cuenta las llamadas posteriores.
- **Repetición.** Toma varias repeticiones y informe el mínimo (menos interferencia)
  o el mediano (típico); diga qué repeticiones.
- **Ámbito.** Para un GPU: el reloj incluyó la copia de datos al y desde el
  dispositivo? Esperó que el GPU terminara (`sincronizar`) antes de detener el
  reloj? Una llamada de GPU devuelve **antes** de que el trabajo esté hecho.
- **El mismo experimento.** La misma máquina, el mismo tamaño de problema, el mismo
  `dtype`, el mismo número de pasos. De lo contrario, está comparando dos cosas diferentes.

*Recuerde: calienta, repite, sincronice, y diga qué contenía el reloj.*

## Cuatro preguntas para comprobarse a sí mismo

1. ¿Cuál corte selecciona la celda **abajo** de cada celda interior de `u`?
2. Si `v = u[1:-1, 1:-1]` y hace `v[:] = 0`, ¿`u` cambia?
3. Un programa es 25% serial. ¿Cuál es su speedup con 4 núcleos? Con núcleos infinitos?
4. Tiemblas `y = f(x_gpu)` y obtienes 0.01 ms. ¿Qué probablemente está mal?

Respuestas: 1. `u[2:, 1:-1]`. 2. Sí; `v` es una vista. 3. `1/(0.25+0.75/4) = 2.29`;
no más de 4. 4. El reloj se detuvo antes de que el GPU terminara; sincronice primero.
