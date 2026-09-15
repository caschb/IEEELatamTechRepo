# Paquete de preparación (45 a 60 minutos)

Realiza esto antes de la sesión, idealmente unos días antes para tener tiempo de solucionar problemas. Necesitas un navegador, una cuenta de Google y un computador normal. Una **CPU** Colab runtime es suficiente; no necesitas una GPU para nada aquí, y una prueba de CPU completada es todo lo que requiere la sesión.

## Lo que podrás hacer después

- Abrir un cuaderno de curso en Colab, guardar tu propia copia, ejecutar y editar celdas, instalar paquetes y recuperar después de una reinicio de la sesión de tiempo.
- Reconocer el pequeño programa (un stencil de difusión de calor) que cada sesión de vida construye y escribir una línea de slicing de NumPy para él.
- Explicar, en una frase cada una: qué es una aceleración, por qué una GPU tiene su propia memoria, y qué una medición debería y no debería incluir.

## Requisitos previos

Python básico: definir una función, un bucle `for`, una lista. Si eso está olvidado, primero haz el refrescante [Refrescador de Python y NumPy](python_numpy_refresher.md). No cuenta como hora.

## Haz estos en orden

| N° | Item | Tiempo | He hecho cuando |
|---|---|---:|---|
| 1 | Leer [`primer.md`](primer.md) | 15 min | Puedes responder a las cuatro preguntas al final |
| 2 | [¿Está mi Colab listo?](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_colab_ready.ipynb) | 10-15 min | La última celda imprime `READY para la sesión` y guardaste el cuaderno |
| 3 | [Ejemplo de ejecución](https://colab.research.google.com/github/caschb/IEEELatamTechRepo/blob/main/notebooks/00_stencil_practice.ipynb) | 10-15 min | La celda de chequeo imprime `PASS` y guardaste el cuaderno |
| 4 | [`self_check.md`](self_check.md) | 5-10 min | Leí la explicación para cualquier pregunta que te salió mal |

## Verificación

- [ ] Guardé mi propia copia de cada cuaderno (Archivo > Guardar una copia en Drive).
- [ ] El cuaderno de preparación imprimió `READY para la sesión`.
- [ ] Reinicié la sesión de tiempo y ejecuté todo de nuevo; aún pasó.
- [ ] El chequeo de stencil imprime `PASS`.
- [ ] Copié la línea `STATUS ...` del cuaderno de preparación a algún lugar donde pueda encontrarla.

## Guardar y descargar tu trabajo

Colab guarda tu copia en Drive automáticamente (también Ctrl+S). Para mantener un archivo que el cuaderno escribió (por ejemplo, una tabla de tiempos), abre el panel de **Archivos** en la izquierda, haz clic derecho en el archivo, **Descargar**. Los archivos en la sesión de tiempo desaparecen cuando se elimina la sesión de tiempo; tu copia de la nota en Drive no.

## Dos cosas que sorprenden a la gente

- **Una nueva sesión de tiempo no tiene nada instalado.** La celda de configuración al principio de cada cuaderno instala lo que falta. Rehazla después de cualquier reinicio. Es rápida la segunda vez.
- **Las sesiones de tiempo se desconectan.** Colab termina las sesiones inactivas y impone límites de uso. Nada se pierde que hayas guardado; reinicia desde el principio.

## Si algo no funciona

Se puede compartir la línea `STATUS ...` (o el error), el número de paso y el
navegador utilizado en el canal de ayuda anunciado por la organización, o enviar
esa información al instructor. Los primeros diez minutos del evento están
reservados para resolver los problemas pendientes de configuración.
