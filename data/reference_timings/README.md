# Tablas de tiempo grabado

Los cuadernos 01 y 02 cargan estos archivos cuando la ejecución en vivo no pueda producir
la comparación por sí misma (un entorno de ejecución de CPU para el recorrido de hilos, sin GPU para los
tiempos de dispositivo). Cada archivo comienza con líneas `# key: value` que describen el
entorno que lo produjo, seguidas de una tabla CSV. Los cuadernos imprimen esta metadata junto a los números
para que los resultados grabados nunca sean confundidos con los de ejecución en vivo.

| Archivo | Producido por | Usado por |
|---|---|---|
| `01_threads.csv` | cuaderno 01, "Punto de control 3" | cuaderno 01, sección de escalado de hilos, cuando no se pueden probar más de dos cantidades de hilos |
| `02_gpu.csv` | cuaderno 02, "Punto de control 2" | cuaderno 02, secciones de recorrido de tamaño y transferencia, en modo de relleno de CPU |
| `04_capstone.csv` | cuaderno 04 | solo para referencia del instructor |

## Estado: datos intermedios

Los archivos actuales provienen del equipo del autor (CPU de 32 núcleos, NVIDIA GeForce RTX 4090, registrados en las líneas de
metadatos), **no provienen de Google Colab**. Existen para comprobar de principio a
fin el modo alternativo. La
relevación 4 en `instructor/validation.md` reemplace estos con una ejecución de GPU documentada en Colab:

1. Abra los cuadernos 01 y 02 en Colab en un nuevo entorno de ejecución de GPU, ejecute todas las celdas.
2. Descargue `timings_01_cpu.csv` y `timings_02_gpu.csv` desde la pestaña de Archivos.
3. Copie estos archivos aquí como `01_threads.csv` y `02_gpu.csv`, comita, proporcione.
4. Ejecute nuevamente el cuaderno 02 en un entorno de CPU y confirme que la línea "RECORDED GPU RUN" muestre el
   entorno de Colab y la GPU.

Los cuadernos obtienen los archivos desde la rama `main` de este repositorio a través de HTTPS; el script de validación del autor lee las copias locales.

## Grabación del demostración de GPU

Grabar la ejecución del cuaderno 02 del instructor con GPU (captura de pantalla, 5 a 8 minutos, desde "Cambiar tipo de ejecución" hasta la tabla de tiempos finales) y poner el enlace en `instructor/run_of_show.md`. La grabación es el segundo relleno, después de esta tabla, para los estudiantes sin GPU.
