# Paquete de preparación (45 a 60 minutos)

Complete esta preparación antes de la sesión, de preferencia con algunos días de
antelación. Necesita un navegador, una cuenta de Google y una computadora. Un entorno
de ejecución de Colab con CPU es suficiente; no se requiere GPU.

## Lo que podrás hacer después

- Abrir un notebook del curso en Colab, guardar una copia propia, ejecutar y editar
  celdas, instalar paquetes y continuar después de reiniciar el entorno de ejecución.
- Reconocer el pequeño programa (un cálculo por vecindad de difusión de calor) que cada sesión de vida construye y escribir una línea de slicing de NumPy para él.
- Explicar, en una frase cada una: qué es una aceleración, por qué una GPU tiene su propia memoria, y qué una medición debería y no debería incluir.

## Requisitos previos

Se requiere conocimiento básico de Python: definir una función, un bucle `for` y una
lista. Si necesita repasarlo, consulte primero el [repaso de Python y NumPy](python_numpy_refresher.md).

## Actividades en orden

| N° | Item | Tiempo | He hecho cuando |
|---|---|---:|---|
| 1 | Leer [`primer.md`](primer.md) | 15 min | Puede responder las cuatro preguntas finales |
| 2 | [¿Está mi Colab listo?](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_colab_ready.ipynb) | 10-15 min | La última celda imprime `READY para la sesión` y guardaste el cuaderno |
| 3 | [Ejemplo de ejecución](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_stencil_practice.ipynb) | 10-15 min | La celda de chequeo imprime `PASS` y guardaste el cuaderno |
| 4 | [`self_check.md`](self_check.md) | 5-10 min | Leí la explicación para cualquier pregunta que salió mal |

## Verificación

- [ ] Guardó una copia propia de cada notebook (Archivo > Guardar una copia en Drive).
- [ ] El cuaderno de preparación imprimió `READY para la sesión`.
- [ ] Reiniciaste la sesión de tiempo y ejecutaste todo de nuevo; aún pasó.
- [ ] El chequeo de cálculo por vecindad imprime `PASS`.
- [ ] Guardó la línea `STATUS ...` del notebook de preparación en un lugar accesible.

## Guardar y descargar el trabajo

Colab guarda automáticamente la copia en Drive (también puede usar Ctrl+S). Para
conservar un archivo generado por el notebook, como una tabla de tiempos, abra el
panel **Archivos**, haga clic derecho sobre el archivo y seleccione **Descargar**.
Los archivos del entorno desaparecen cuando este se elimina; la copia en Drive se conserva.

## Dos cosas que sorprenden a la gente

- **Una nueva sesión de tiempo no tiene nada instalado.** La celda de configuración al principio de cada cuaderno instala lo que falta. Rehazla después de cualquier reiniciación. Es rápida la segunda vez.
- **Las sesiones de tiempo se desconectan.** Colab termina las sesiones inactivas y impone límites de uso. Nada se pierde que hayas guardado; reinicia desde el principio.

## Si algo no funciona

Comparte la línea `STATUS ...` (o el error), el número de paso y el navegador utilizado en el canal de ayuda anunciado por la organización, o envía esa información al instructor. Los primeros diez minutos del evento están reservados para resolver los problemas pendientes de configuración.
