import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Análisis científico y de datos con Polars y marimo

    ¡Bienvenidos a este módulo del curso de **HPC con Python**!

    En este cuaderno exploraremos cómo **polars** y **marimo** forman una combinación
    potente para el análisis interactivo de datos científicos. Cubriremos:

    - **Preparación del entorno** — importaciones, carga de datos y configuración
    - **Conjunto de datos público** — sismos en tiempo real del USGS (¡se actualiza cada minuto!)
    - **Análisis exploratorio (EDA)** — con SQL, expresiones de polars y la interfaz reactiva de marimo
    - **Visualizaciones** — gráficos interactivos con Altair
    - **Medición comparativa** — polars contra pandas en las mismas cargas de trabajo, medido con honestidad
    - **Aprendizaje automático** — predicción de la magnitud sísmica con regresión lineal y random forest

    Los datos provienen del **Programa de Riesgos Sísmicos del USGS**, que publica un
    flujo en vivo de todos los eventos sísmicos del mundo durante los últimos 30 días.
    Es un buen ejemplo de datos científicos reales: vienen sucios, son multidimensionales
    y se benefician tanto de consultas tipo SQL como de la visualización.
    """)
    return


@app.cell
def _():
    # --- Preparación: volver a ejecutar tras cada reinicio del entorno ----------
    # Misma convención que el resto de los cuadernos del curso: primero importar, y
    # solo instalar con pip lo que realmente falte. En molab y en una instalación
    # local de marimo esto no suele hacer nada; en Colab instala polars y marimo.
    import importlib
    import importlib.util
    import time

    def asegurar(modulo, paquete=None):
        """Importa `modulo`, instalando `paquete` con pip solo si la importación falla.

        `subprocess`/`sys` se importan dentro de la función a propósito: marimo exige
        que cada nombre de nivel superior se defina en exactamente una celda, y la
        celda de detección de hardware de la Parte 3 también necesita `subprocess`.
        """
        if importlib.util.find_spec(modulo) is None:
            import subprocess as _sp
            import sys as _sys

            _sp.run(
                [_sys.executable, "-m", "pip", "install", "-q", paquete or modulo],
                check=True,
            )
        return importlib.import_module(modulo)

    mo = asegurar("marimo")
    np = asegurar("numpy")
    pd = asegurar("pandas")
    pl = asegurar("polars")
    alt = asegurar("altair")
    asegurar("sklearn", "scikit-learn")
    asegurar("pyarrow")          # intercambio polars <-> pandas
    # inspección de los grupos de hilos nativos en la Parte 3
    asegurar("threadpoolctl")
    asegurar("psutil")           # detección de núcleos físicos y memoria RAM

    mo.md(
        "✅ Entorno listo — polars, pandas, numpy, marimo, altair y "
        "scikit-learn están cargados."
    )
    return alt, mo, np, pd, pl, time


@app.cell
def _(mo, pl):
    # Carga del flujo de sismos del USGS (todos los sismos, últimos 30 días, actualizado automáticamente)
    SISMOS_CSV = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_month.csv"

    sismos = pl.read_csv(SISMOS_CSV, try_parse_dates=True)

    mo.md(f"""
    **Datos cargados:** sismos del USGS (últimos 30 días)

    | Propiedad | Valor |
    |---|---|
    | Filas | {sismos.height:,} |
    | Columnas | {sismos.width} |
    | Memoria | {sismos.estimated_size('mb'):.1f} MB |
    | Esquema | {sismos.columns} |
    """)
    return (sismos,)


@app.cell
def _(mo, sismos):
    _df = mo.sql(
        f"""
        SELECT time, place, mag, depth, latitude, longitude, type
        FROM sismos
        ORDER BY mag DESC
        LIMIT 15
        """
    )
    return


@app.cell
def _(pl, sismos):
    # Estadísticas descriptivas de las columnas numéricas principales
    resumen = sismos.select([
        pl.col("mag").alias("magnitud"),
        pl.col("depth").alias("profundidad"),
        pl.col("gap").alias("brecha_azimutal"),
        pl.col("rms").alias("rms"),
    ]).describe()

    resumen
    return


@app.cell
def _(mo, sismos):
    # Controles interactivos de marimo para filtrar
    filtro_mag = mo.ui.slider(
        start=0, stop=10, step=0.1, value=2.5, label="Magnitud mínima"
    )
    filtro_tipo = mo.ui.dropdown(
        options=sorted(sismos["type"].unique().to_list()),
        value="earthquake",
        label="Tipo de evento",
    )
    filtro_prof = mo.ui.slider(
        start=0, stop=700, step=10, value=0, label="Profundidad máxima (km)"
    )

    mo.hstack([filtro_mag, filtro_tipo, filtro_prof])
    return filtro_mag, filtro_prof, filtro_tipo


@app.cell(hide_code=True)
def _(filtro_mag, filtro_prof, filtro_tipo, mo, pl, sismos):
    # Aplicar los filtros según lo elegido en la interfaz
    sismos_filtrados = sismos.filter(
        (pl.col("mag") >= filtro_mag.value)
        & (pl.col("type") == filtro_tipo.value)
        & (pl.col("depth") <= filtro_prof.value)
    )

    mo.md(f"""
    ### Conjunto de datos filtrado

    Se muestran **{sismos_filtrados.height:,}** eventos con:

    - Magnitud ≥ **{filtro_mag.value}**
    - Tipo = **{filtro_tipo.value}**
    - Profundidad ≤ **{filtro_prof.value} km**
    """)
    return (sismos_filtrados,)


@app.cell
def _(mo, sismos_filtrados):
    _df = mo.sql(
        f"""
        SELECT
            DATE(time) AS fecha,
            COUNT(*) AS cantidad_eventos,
            ROUND(AVG(mag), 2) AS magnitud_promedio,
            ROUND(MAX(mag), 1) AS magnitud_maxima
        FROM sismos_filtrados
        GROUP BY DATE(time)
        ORDER BY fecha
        """
    )
    return


@app.cell
def _(alt, sismos_filtrados):
    # Gráfico de Altair: distribución de magnitudes
    grafico_hist = alt.Chart(sismos_filtrados.to_pandas()).mark_bar(
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        x=alt.X("mag:Q", bin=alt.Bin(maxbins=40), title="Magnitud"),
        y=alt.Y("count():Q", title="Cantidad de eventos"),
        tooltip=["count():Q", "mag:Q"],
    ).properties(
        title="Distribución de magnitudes sísmicas",
        width=600,
        height=300,
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_title(fontSize=16)

    grafico_hist
    return


@app.cell
def _(alt, sismos_filtrados):
    # Gráfico de Altair: dispersión magnitud contra profundidad (coloreado por magnitud)
    grafico_dispersion = alt.Chart(sismos_filtrados.to_pandas()).mark_circle(opacity=0.6).encode(
        x=alt.X("depth:Q", title="Profundidad (km)",
                scale=alt.Scale(zero=True)),
        y=alt.Y("mag:Q", title="Magnitud", scale=alt.Scale(zero=True)),
        color=alt.Color("mag:Q", scale=alt.Scale(
            scheme="viridis"), title="Magnitud"),
        size=alt.Size("mag:Q", scale=alt.Scale(
            range=[10, 200]), title="Magnitud"),
        tooltip=["place:N", "mag:Q", "depth:Q", "time:T"],
    ).properties(
        title="Magnitud contra profundidad",
        width=600,
        height=400,
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_title(fontSize=16)

    grafico_dispersion
    return


@app.cell
def _(alt, pl, sismos_filtrados):
    # Conteos diarios a partir de los sismos filtrados
    conteos_diarios = (
        sismos_filtrados
        .with_columns(pl.col("time").cast(pl.Date).alias("fecha"))
        .group_by("fecha")
        .agg(
            pl.len().alias("cantidad_eventos"),
            pl.col("mag").mean().round(2).alias("magnitud_promedio"),
            pl.col("mag").max().round(1).alias("magnitud_maxima"),
        )
        .sort("fecha")
    )

    # Gráfico de Altair: actividad sísmica diaria
    grafico_linea_tiempo = alt.Chart(conteos_diarios).mark_bar(
        cornerRadiusTopLeft=2,
        cornerRadiusTopRight=2,
    ).encode(
        x=alt.X("fecha:T", title="Fecha"),
        y=alt.Y("cantidad_eventos:Q", title="Cantidad de eventos"),
        color=alt.Color(
            "magnitud_maxima:Q",
            scale=alt.Scale(scheme="reds"),
            title="Magnitud máxima",
        ),
        tooltip=[
            "fecha:T",
            "cantidad_eventos:Q",
            "magnitud_promedio:Q",
            "magnitud_maxima:Q",
        ],
    ).properties(
        title="Actividad sísmica diaria (últimos 30 días)",
        width=600,
        height=300,
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14,
    ).configure_title(fontSize=16)

    grafico_linea_tiempo
    return


@app.cell
def _(mo, sismos_filtrados):
    # Tabla con los 10 sismos más fuertes
    mas_fuertes = (
        sismos_filtrados
        .select(["time", "place", "mag", "depth", "latitude", "longitude"])
        .sort("mag", descending=True)
        .head(10)
    )

    mo.ui.table(mas_fuertes, selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Parte 2 — Comparación entre polars y pandas

    pandas ha sido la biblioteca de dataframes por defecto en Python durante más de una
    década, y polars es el contendiente más reciente, escrito en Rust. En lugar de
    confiar en cifras de propaganda, vamos a **medir ambas con las mismas cargas de
    trabajo y los mismos datos**.

    Dos condiciones hacen que la comparación sea justa y significativa:

    1. **Mismo trabajo, misma máquina.** Cada operación se ejecuta sobre datos idénticos
       en el mismo proceso, de modo que comparamos implementaciones y no hardware.
    2. **Datos suficientes para que importe.** El flujo en vivo tiene apenas ~10 mil
       filas, tan poco que los tiempos quedan dominados por el intérprete y el ruido de
       medición. Lo replicamos hasta unos cientos de miles de filas, que es donde las
       diferencias de arquitectura (multihilo, memoria Arrow, optimización de consultas)
       realmente se manifiestan.

    Esta es la lección central de HPC del módulo: **medir antes de optimizar**, y
    asegurarse de que la medición sea lo bastante grande como para medir lo que importa.
    """)
    return


@app.cell
def _(mo):
    # Control del tamaño de la prueba: cuántas copias del flujo se apilan.
    factor_escala = mo.ui.slider(
        start=5, stop=60, step=5, value=40, label="Escala de datos (copias del flujo)"
    )
    repeticiones = mo.ui.slider(
        start=1, stop=7, step=1, value=3, label="Repeticiones por medición"
    )

    mo.hstack([factor_escala, repeticiones])
    return factor_escala, repeticiones


@app.cell
def _(factor_escala, mo, pl, sismos):
    import tempfile
    from pathlib import Path

    # Construimos un conjunto mayor apilando copias del flujo en vivo y lo escribimos
    # a disco una sola vez. Así ambas bibliotecas leen exactamente el mismo archivo
    # desde la misma caché de páginas del sistema operativo.
    dir_pruebas = Path(tempfile.gettempdir()) / "pruebas_polars"
    dir_pruebas.mkdir(exist_ok=True)

    marco_grande = pl.concat([sismos] * factor_escala.value)
    ruta_csv = dir_pruebas / "sismos_grande.csv"
    ruta_parquet = dir_pruebas / "sismos_grande.parquet"

    marco_grande.write_csv(ruta_csv)
    marco_grande.write_parquet(ruta_parquet)

    mb_csv = ruta_csv.stat().st_size / 1e6
    mb_parquet = ruta_parquet.stat().st_size / 1e6

    mo.md(f"""
    ### Datos de prueba preparados

    | Propiedad | Valor |
    |---|---|
    | Filas | {marco_grande.height:,} |
    | Columnas | {marco_grande.width} |
    | CSV en disco | {mb_csv:.1f} MB |
    | Parquet en disco | {mb_parquet:.1f} MB |
    | Razón de compresión de Parquet | {mb_csv / mb_parquet:.1f}× más pequeño |

    Nótese ya la diferencia de tamaño: **Parquet es columnar y comprimido**, así que
    ocupa una fracción del CSV en disco y exige mucha menos entrada/salida para leerse.
    Elegir el formato de archivo correcto suele rendir más que elegir la biblioteca
    correcta.
    """)
    return ruta_csv, ruta_parquet


@app.cell
def _(repeticiones, time):
    def medir(fn, n_repeticiones):
        """Ejecuta `fn` n veces y devuelve el mejor tiempo de reloj, en segundos.

        Reportamos el *mínimo*, no el promedio: la corrida más rápida es la menos
        perturbada por el planificador del sistema operativo, pausas del recolector
        de basura y demás ruido. Es la misma convención que usa `timeit`.
        """
        tiempos = []
        for _ in range(n_repeticiones):
            inicio = time.perf_counter()
            fn()
            tiempos.append(time.perf_counter() - inicio)
        return min(tiempos)

    def comparar(etiqueta, fn_polars, fn_pandas, n_repeticiones):
        t_polars = medir(fn_polars, n_repeticiones)
        t_pandas = medir(fn_pandas, n_repeticiones)
        return {
            "operacion": etiqueta,
            "polars_s": round(t_polars, 4),
            "pandas_s": round(t_pandas, 4),
            "aceleracion": round(t_pandas / t_polars, 2),
        }

    n_repeticiones = repeticiones.value
    return comparar, n_repeticiones


@app.cell
def _(comparar, mo, n_repeticiones, pd, pl, ruta_csv, ruta_parquet):
    # --- Medición 1: lectura de archivos --------------------------------------------
    # polars analiza el CSV usando todos los núcleos; pandas lo hace en un solo núcleo.
    resultados_lectura = [
        comparar(
            "Leer CSV",
            lambda: pl.read_csv(ruta_csv, try_parse_dates=True),
            lambda: pd.read_csv(ruta_csv, parse_dates=["time", "updated"]),
            n_repeticiones,
        ),
        comparar(
            "Leer Parquet",
            lambda: pl.read_parquet(ruta_parquet),
            lambda: pd.read_parquet(ruta_parquet),
            n_repeticiones,
        ),
    ]

    tabla_lectura = pl.DataFrame(resultados_lectura)

    mo.vstack([
        mo.md("### Medición 1 — Entrada y salida de archivos"),
        mo.ui.table(tabla_lectura, selection=None, pagination=False),
        mo.md(
            "El análisis de CSV es donde polars gana por más margen: usa un **analizador "
            "multihilo y acelerado con SIMD** escrito en Rust, mientras que pandas lo hace "
            "en un solo núcleo. Parquet reduce bastante la diferencia, porque ambas "
            "bibliotecas leen un formato binario columnar y hacen mucho menos trabajo por "
            "cada byte."
        ),
    ])
    return


@app.cell
def _(pd, pl, ruta_csv):
    # Cargamos ambos marcos una sola vez para que las mediciones siguientes midan
    # cómputo y no entrada/salida.
    pl_df = pl.read_csv(ruta_csv, try_parse_dates=True)
    pd_df = pd.read_csv(ruta_csv, parse_dates=["time", "updated"])
    return pd_df, pl_df


@app.cell
def _(comparar, mo, n_repeticiones, pd_df, pl, pl_df):
    # --- Medición 2: filtrado + agrupación + agregación ------------------------------
    def polars_agrupar():
        return (
            pl_df.filter(pl.col("mag") >= 2.5)
            .group_by("type")
            .agg(
                pl.len().alias("n"),
                pl.col("mag").mean().alias("mag_promedio"),
                pl.col("depth").max().alias("prof_maxima"),
            )
        )

    def pandas_agrupar():
        return (
            pd_df[pd_df["mag"] >= 2.5]
            .groupby("type")
            .agg(
                n=("mag", "size"),
                mag_promedio=("mag", "mean"),
                prof_maxima=("depth", "max"),
            )
        )

    # --- Medición 3: ordenamiento ----------------------------------------------------
    def polars_ordenar():
        return pl_df.sort("mag", descending=True)

    def pandas_ordenar():
        return pd_df.sort_values("mag", ascending=False)

    # --- Medición 4: filtrado de cadenas de texto ------------------------------------
    def polars_texto():
        return pl_df.filter(pl.col("place").str.contains("California")).height

    def pandas_texto():
        return len(pd_df[pd_df["place"].str.contains("California", na=False)])

    # --- Medición 5: columnas derivadas ----------------------------------------------
    def polars_derivar():
        return pl_df.with_columns(
            (10 ** (1.5 * pl.col("mag") + 9.1)).alias("energia_joules"),
            (pl.col("depth") / 6371.0).alias("fraccion_profundidad"),
        )

    def pandas_derivar():
        salida = pd_df.copy()
        salida["energia_joules"] = 10 ** (1.5 * salida["mag"] + 9.1)
        salida["fraccion_profundidad"] = salida["depth"] / 6371.0
        return salida

    resultados_computo = [
        comparar("Filtrar + agrupar + agregar", polars_agrupar,
                 pandas_agrupar, n_repeticiones),
        comparar("Ordenar por magnitud", polars_ordenar,
                 pandas_ordenar, n_repeticiones),
        comparar("Filtrar por subcadena", polars_texto,
                 pandas_texto, n_repeticiones),
        comparar("Columnas derivadas (matemáticas)",
                 polars_derivar, pandas_derivar, n_repeticiones),
    ]

    tabla_computo = pl.DataFrame(resultados_computo)

    mo.vstack([
        mo.md("### Mediciones 2 a 5 — Cómputo en memoria"),
        mo.ui.table(tabla_computo, selection=None, pagination=False),
    ])
    return (resultados_computo,)


@app.cell
def _(alt, pl, resultados_computo):
    # Visualizamos las aceleraciones. Un valor de 1.0 significa "no hay diferencia".
    df_aceleracion = pl.DataFrame(resultados_computo).select([
        "operacion", "aceleracion"])

    grafico_aceleracion = (
        alt.Chart(df_aceleracion)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("aceleracion:Q",
                    title="Aceleración (tiempo de pandas ÷ tiempo de polars)"),
            y=alt.Y("operacion:N", title=None, sort="-x"),
            color=alt.Color("aceleracion:Q", scale=alt.Scale(
                scheme="blues"), legend=None),
            tooltip=["operacion:N", "aceleracion:Q"],
        )
        .properties(
            title="Aceleración de polars sobre pandas (más alto = polars más rápido)",
            width=600,
            height=200,
        )
    )

    # Línea de referencia en 1.0: a la izquierda de ella, pandas fue más rápido.
    linea_base = (
        alt.Chart(pl.DataFrame({"x": [1.0]}))
        .mark_rule(color="firebrick", strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    (grafico_aceleracion + linea_base).configure_axis(
        labelFontSize=12, titleFontSize=14
    ).configure_title(fontSize=16)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ¿Por qué polars es más rápido?

    Las aceleraciones no son magia, y tampoco son uniformes. Provienen de decisiones
    concretas de ingeniería que conviene conocer como practicantes de HPC:

    | Mecanismo | Qué significa |
    |---|---|
    | **Multihilo por defecto** | polars satura todos los núcleos sin que uno escriba código paralelo. pandas es de un solo hilo en la mayoría de sus operaciones. |
    | **Disposición de memoria Apache Arrow** | Columnar, amigable con la caché y sin copias entre herramientas. Los bucles vectorizados fallan menos en caché. |
    | **Sin índice** | pandas mantiene un índice de filas y alinea por él; polars elimina ese concepto y evita toda una clase de trabajo oculto. |
    | **Rust + SIMD** | Los núcleos de cómputo están compilados, no interpretados, y usan instrucciones vectoriales. |
    | **Evaluación diferida** | `scan_csv` y `LazyFrame` permiten que un optimizador reordene y fusione operaciones, y que omita columnas por completo. |
    """)
    return


@app.cell
def _(mo, n_repeticiones, pl, ruta_csv, time):
    # --- Evaluación diferida: la idea más importante de polars para HPC -------------
    # Ansiosa: lee todas las columnas y filas a memoria, y luego descarta casi todo.
    def flujo_ansioso():
        return (
            pl.read_csv(ruta_csv, try_parse_dates=True)
            .filter(pl.col("mag") >= 4.0)
            .group_by("type")
            .agg(pl.col("mag").mean().alias("mag_promedio"))
        )

    # Diferida: describe primero la consulta y deja que el optimizador empuje el filtro
    # y la selección de columnas hacia dentro del escaneo del CSV, de modo que los datos
    # innecesarios nunca lleguen a materializarse.
    def flujo_diferido():
        return (
            pl.scan_csv(ruta_csv, try_parse_dates=True)
            .filter(pl.col("mag") >= 4.0)
            .group_by("type")
            .agg(pl.col("mag").mean().alias("mag_promedio"))
            .collect()
        )

    def _medir(fn):
        tiempos = []
        for _ in range(n_repeticiones):
            inicio = time.perf_counter()
            fn()
            tiempos.append(time.perf_counter() - inicio)
        return min(tiempos)

    t_ansioso = _medir(flujo_ansioso)
    t_diferido = _medir(flujo_diferido)

    mo.md(f"""
    ### Extra — Evaluación ansiosa contra diferida en polars

    Mismo resultado, mismo archivo, dos estrategias de ejecución:

    | Estrategia | Tiempo |
    |---|---|
    | Ansiosa (Eager) (`read_csv` → filtrar → agrupar) | {t_ansioso:.3f} s |
    | Diferida (Deferred) (`scan_csv` → filtrar → agrupar → `collect`) | {t_diferido:.3f} s |
    | **Aceleración por diferir** | **{t_ansioso / t_diferido:.2f}×** |

    La versión diferida gana porque `collect()` dispara un optimizador que aplica
    **empuje de proyección** (leer solo las columnas que la consulta menciona) y
    **empuje de predicados** (descartar filas durante el escaneo, antes de que lleguen a
    materializarse). Esta es exactamente la estrategia fuera de memoria que se necesita
    cuando los datos superan la RAM disponible: el patrón escala a archivos que jamás
    podrían cargarse con `read_csv`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### El plan de consulta

    Se puede inspeccionar lo que decidió el optimizador antes de ejecutar nada. Es el
    equivalente, en dataframes, a leer la salida de un compilador, y es la forma de
    depurar un flujo lento.
    """)
    return


@app.cell
def _(pl, ruta_csv):
    # `explain()` muestra el plan optimizado sin ejecutarlo.
    plan_consulta = (
        pl.scan_csv(ruta_csv, try_parse_dates=True)
        .filter(pl.col("mag") >= 4.0)
        .select(["type", "mag", "depth"])
        .group_by("type")
        .agg(pl.col("mag").mean().alias("mag_promedio"))
    )

    print(plan_consulta.explain())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    # Parte 3 — Aprendizaje automático: predecir la magnitud sísmica

    Ahora ponemos los datos a un uso científico. **Pregunta:** ¿se puede predecir la
    magnitud de un evento sísmico a partir de la geometría y la calidad de las
    observaciones de la red sismológica?

    El flujo del USGS trae varias columnas que describen *cómo se midió el evento*, no
    solo dónde ocurrió:

    | Variable | Significado |
    |---|---|
    | `depth` | Profundidad del hipocentro, en km |
    | `nst` | Número de estaciones usadas para localizar el evento |
    | `gap` | Mayor brecha azimutal entre estaciones (grados); mientras menor, mejor restringida está la localización |
    | `dmin` | Distancia horizontal a la estación más cercana, en grados |
    | `rms` | Residuo cuadrático medio del ajuste de tiempos de viaje |
    | `horizontalError`, `depthError` | Estimaciones de incertidumbre de la localización, en km |
    | `magNst` | Número de estaciones usadas para calcular la magnitud |
    | `latitude`, `longitude` | Dónde ocurrió el evento |

    **Una advertencia científica importante, dicha de entrada.** Estas variables las
    registra el *mismo proceso de análisis* que produce la magnitud. En particular `dmin`
    y `nst` reflejan qué estaciones llegaron a detectar el evento, y los sismos más
    grandes son detectados por más estaciones y desde más lejos. Por lo tanto esto es una
    tarea de **reconstrucción**, no de pronóstico: estamos aprendiendo cómo se relacionan
    entre sí las salidas del proceso de medición, no prediciendo sismos futuros. Esa
    distinción importa, y confundir ambas cosas es una de las fallas más comunes al
    aplicar aprendizaje automático a datos científicos.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.1 — ¿Qué recursos de CPU usa realmente scikit-learn?

    Antes de medir algo, conviene saber *sobre qué* lo estamos midiendo. scikit-learn no
    anuncia su propio paralelismo, así que esta celda lo inspecciona directamente:

    - **Núcleos lógicos contra físicos.** `n_jobs=-1` usa los núcleos lógicos (los
      hyperthreads). Para trabajo numérico limitado por cómputo, los hyperthreads
      comparten unidades de ejecución, así que la aceleración útil suele seguir el
      número de núcleos *físicos*.
    - **Grupos de hilos nativos.** NumPy y SciPy llaman a BLAS (OpenBLAS o MKL), y los
      núcleos en Cython de scikit-learn usan OpenMP. Estos grupos son independientes de
      `n_jobs`, y **se multiplican entre sí**: `n_jobs=8` con 8 hilos de BLAS cada uno
      puede sobresuscribir la máquina a 64 hilos y resultar *más lento*. `threadpoolctl`
      es la forma de verlos y controlarlos.

    scikit-learn es una biblioteca **de CPU**, así que todo lo que sigue se mide en CPU.
    La aceleración por GPU para aprendizaje automático clásico se trata aparte, en la
    Parte 4 con RAPIDS.
    """)
    return


@app.cell
def _(mo, np):
    import os
    import platform

    import joblib
    import sklearn
    import threadpoolctl

    # Los núcleos físicos requieren psutil; si no está, se degrada con elegancia.
    try:
        import psutil

        nucleos_fisicos = psutil.cpu_count(logical=False)
        ram_gb = round(psutil.virtual_memory().total / 2**30, 1)
    except ImportError:
        nucleos_fisicos, ram_gb = None, None

    nucleos_logicos = os.cpu_count()

    # Los grupos de hilos nativos que están por debajo de scikit-learn.
    grupos = threadpoolctl.threadpool_info()

    HARDWARE = {
        "entorno": platform.node(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
        "nucleos_logicos": nucleos_logicos,
        "nucleos_fisicos": nucleos_fisicos if nucleos_fisicos else "desconocido",
        "ram_gb": ram_gb if ram_gb else "desconocido",
    }

    filas_grupos = "\n".join(
        f"| `{g.get('internal_api')}` | {g.get('num_threads')} | {g.get('version') or '—'} |"
        for g in grupos
    ) or "| — | — | — |"

    mo.md(f"""
    ### Recursos de CPU visibles para scikit-learn

    | Propiedad | Valor |
    |---|---|
    | Entorno | {HARDWARE["entorno"]} |
    | Python | {HARDWARE["python"]} |
    | scikit-learn | {HARDWARE["sklearn"]} |
    | joblib | {HARDWARE["joblib"]} |
    | **Núcleos lógicos** (los que usa `n_jobs=-1`) | **{HARDWARE["nucleos_logicos"]}** |
    | **Núcleos físicos** (los que realmente escalan) | **{HARDWARE["nucleos_fisicos"]}** |
    | RAM | {HARDWARE["ram_gb"]} GB |

    #### Grupos de hilos nativos por debajo de scikit-learn

    | Backend | Hilos | Versión |
    |---|---|---|
    {filas_grupos}

    Cada uno de estos grupos se controla de forma independiente de `n_jobs`. Cuando un
    trabajo paralelo se vuelve *más lento* al agregar trabajadores, la sobresuscripción
    de estos grupos suele ser la causa.
    """)
    return (HARDWARE,)


@app.cell
def _(mo, pl, sismos):
    VARIABLES = [
        "depth", "nst", "gap", "dmin", "rms",
        "horizontalError", "depthError", "magNst",
        "latitude", "longitude",
    ]
    OBJETIVO = "mag"

    # Conservamos solo sismos tectónicos: las voladuras de cantera y las explosiones son
    # un proceso físico distinto, y mezclarlas difuminaría lo que el modelo aprende.
    datos_ml = (
        sismos.filter(pl.col("type") == "earthquake")
        .select(VARIABLES + [OBJETIVO])
        .drop_nulls()
    )

    descartadas = sismos.filter(
        pl.col("type") == "earthquake").height - datos_ml.height

    mo.md(f"""
    ### Matriz de variables preparada

    | Propiedad | Valor |
    |---|---|
    | Sismos disponibles | {sismos.filter(pl.col("type") == "earthquake").height:,} |
    | Filas descartadas (valores faltantes) | {descartadas:,} |
    | **Filas utilizables para entrenar** | **{datos_ml.height:,}** |
    | Variables | {len(VARIABLES)} |
    | Objetivo | `{OBJETIVO}` (magnitud de momento o local) |

    Descartamos las filas con valores faltantes en lugar de imputarlos. Con esta cantidad
    de datos restantes es la decisión honesta: imputar `dmin` o `gap` sería inventar una
    geometría de red que nunca se observó.
    """)
    return OBJETIVO, VARIABLES, datos_ml


@app.cell
def _(OBJETIVO, VARIABLES, datos_ml):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    # El paso de polars a numpy es una entrega casi sin copias; scikit-learn espera
    # arreglos de numpy.
    X = datos_ml.select(VARIABLES).to_numpy()
    y = datos_ml[OBJETIVO].to_numpy()

    X_entrena, X_prueba, y_entrena, y_prueba = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    return (
        LinearRegression,
        RandomForestRegressor,
        StandardScaler,
        X,
        X_entrena,
        X_prueba,
        cross_val_score,
        make_pipeline,
        mean_absolute_error,
        r2_score,
        y,
        y_entrena,
        y_prueba,
    )


@app.cell
def _(time):
    def entrenar_con_tiempo(modelo, X_tr, y_tr, X_te, n_repeticiones=3):
        """Entrena y predice; devuelve (modelo, predicciones, s_entrenamiento, s_prediccion).

        Igual que en la Parte 2, reportamos el *mínimo* de varias repeticiones. El
        entrenamiento se rehace desde cero en cada repetición, de modo que no se arrastra
        estado tibio entre corridas.
        """
        tiempos_entrena, tiempos_predice = [], []
        for _ in range(n_repeticiones):
            t0 = time.perf_counter()
            modelo.fit(X_tr, y_tr)
            tiempos_entrena.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            predicciones = modelo.predict(X_te)
            tiempos_predice.append(time.perf_counter() - t0)
        return modelo, predicciones, min(tiempos_entrena), min(tiempos_predice)

    return (entrenar_con_tiempo,)


@app.cell
def _(
    LinearRegression,
    StandardScaler,
    X_entrena,
    X_prueba,
    entrenar_con_tiempo,
    make_pipeline,
    mean_absolute_error,
    mo,
    np,
    r2_score,
    y_entrena,
    y_prueba,
):
    # --- Modelo 1: regresión lineal --------------------------------------------------
    # Escalar importa aquí: las variables tienen unidades muy distintas (grados, km,
    # conteos). StandardScaler las pone en pie de igualdad para que los coeficientes
    # sean interpretables como importancia relativa.
    modelo_lineal = make_pipeline(StandardScaler(), LinearRegression())

    modelo_lineal, pred_lineal, s_entrena_lineal, s_predice_lineal = entrenar_con_tiempo(
        modelo_lineal, X_entrena, y_entrena, X_prueba
    )

    r2_lineal = r2_score(y_prueba, pred_lineal)
    mae_lineal = mean_absolute_error(y_prueba, pred_lineal)

    # Referencia ingenua: predecir siempre el promedio de entrenamiento. Cualquier
    # modelo útil debe superarla.
    mae_base = np.mean(np.abs(y_prueba - y_entrena.mean()))

    mo.md(f"""
    ### Modelo 1 — Regresión lineal

    | Métrica | Valor |
    |---|---|
    | R² (varianza explicada) | **{r2_lineal:.3f}** |
    | Error absoluto medio | **{mae_lineal:.3f}** unidades de magnitud |
    | MAE de referencia (predecir el promedio) | {mae_base:.3f} |
    | Mejora sobre la referencia | {(1 - mae_lineal / mae_base) * 100:.0f}% |
    | **Tiempo de entrenamiento** | **{s_entrena_lineal * 1000:.1f} ms** |
    | **Tiempo de predicción** | **{s_predice_lineal * 1000:.2f} ms** |

    Una recta a través de 10 variables explica cerca del {r2_lineal * 100:.0f}% de la
    varianza de la magnitud: un resultado sólido para un modelo tan simple, y bastante
    mejor que adivinar el promedio.

    Nótese el costo de entrenamiento: **{s_entrena_lineal * 1000:.1f} ms**. Ajustar
    mínimos cuadrados ordinarios es una única resolución en forma cerrada, así que es
    prácticamente gratis con esta cantidad de datos. Conviene tener ese número presente
    cuando entrene el bosque más abajo.
    """)
    return (
        mae_lineal,
        modelo_lineal,
        pred_lineal,
        r2_lineal,
        s_entrena_lineal,
        s_predice_lineal,
    )


@app.cell
def _(VARIABLES, alt, mo, modelo_lineal, pl):
    # Como estandarizamos las variables, los coeficientes son directamente comparables:
    # cada uno es el cambio en magnitud por una desviación estándar de esa variable.
    coeficientes = (
        pl.DataFrame({
            "variable": VARIABLES,
            "coeficiente": modelo_lineal[-1].coef_,
        })
        .with_columns(pl.col("coeficiente").abs().alias("magnitud_del_efecto"))
        .sort("magnitud_del_efecto", descending=True)
    )

    grafico_coef = (
        alt.Chart(coeficientes)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("coeficiente:Q",
                    title="Coeficiente (por desviación estándar)"),
            y=alt.Y("variable:N", title=None, sort="-x"),
            color=alt.condition(
                alt.datum.coeficiente > 0,
                alt.value("#2b6cb0"),
                alt.value("#c53030"),
            ),
            tooltip=["variable:N", "coeficiente:Q"],
        )
        .properties(
            title="Coeficientes de la regresión lineal (azul = aumenta la magnitud)",
            width=600,
            height=280,
        )
    )

    mo.vstack([
        mo.md("#### Qué aprendió el modelo lineal"),
        grafico_coef,
        mo.md(
            "Las barras azules empujan la magnitud predicha hacia arriba; las rojas, hacia "
            "abajo. `rms` y `nst` son las de mayor peso: los eventos con residuos de tiempo "
            "de viaje más grandes y con más estaciones reportando tienden a ser mayores, "
            "que es justamente el efecto de detección señalado arriba, visible de forma "
            "directa en los coeficientes del modelo."
        ),
    ])
    return


@app.cell
def _(
    RandomForestRegressor,
    X_entrena,
    X_prueba,
    entrenar_con_tiempo,
    mean_absolute_error,
    mo,
    r2_score,
    y_entrena,
    y_prueba,
):
    # --- Modelo 2: random forest -----------------------------------------------------
    # n_jobs=-1 entrena los árboles usando todos los núcleos: la conexión con HPC es que
    # los ensambles son vergonzosamente paralelos, así que esto escala casi linealmente
    # con la cantidad de núcleos.
    modelo_bosque = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    modelo_bosque, pred_bosque, s_entrena_bosque, s_predice_bosque = entrenar_con_tiempo(
        modelo_bosque, X_entrena, y_entrena, X_prueba
    )

    r2_bosque = r2_score(y_prueba, pred_bosque)
    mae_bosque = mean_absolute_error(y_prueba, pred_bosque)

    mo.md(f"""
    ### Modelo 2 — Random forest

    | Métrica | Valor |
    |---|---|
    | R² (varianza explicada) | **{r2_bosque:.3f}** |
    | Error absoluto medio | **{mae_bosque:.3f}** unidades de magnitud |
    | **Tiempo de entrenamiento** (200 árboles, todos los núcleos) | **{s_entrena_bosque:.3f} s** |
    | **Tiempo de predicción** | **{s_predice_bosque * 1000:.1f} ms** |

    El bosque captura estructura **no lineal** e **interacciones entre variables** que una
    recta no puede: el efecto de `dmin` sobre la magnitud no es una pendiente constante, y
    además depende de `nst`. Los árboles modelan eso de forma natural.

    Obsérvese `n_jobs=-1` en el constructor: cada árbol se entrena de forma independiente,
    así que un random forest es *vergonzosamente paralelo* y escala casi linealmente con
    los núcleos disponibles. Es uno de los ejemplos más claros de paralelismo estilo HPC
    en el trabajo cotidiano de aprendizaje automático.
    """)
    return (
        mae_bosque,
        modelo_bosque,
        pred_bosque,
        r2_bosque,
        s_entrena_bosque,
        s_predice_bosque,
    )


@app.cell
def _(
    mae_bosque,
    mae_lineal,
    mo,
    pl,
    r2_bosque,
    r2_lineal,
    s_entrena_bosque,
    s_entrena_lineal,
    s_predice_bosque,
    s_predice_lineal,
):
    comparacion = pl.DataFrame({
        "modelo": ["Regresión lineal", "Random forest"],
        "r2": [round(r2_lineal, 3), round(r2_bosque, 3)],
        "mae": [round(mae_lineal, 3), round(mae_bosque, 3)],
        "s_entrenamiento": [round(s_entrena_lineal, 4), round(s_entrena_bosque, 4)],
        "s_prediccion": [round(s_predice_lineal, 5), round(s_predice_bosque, 5)],
    })

    _costo = s_entrena_bosque / \
        s_entrena_lineal if s_entrena_lineal > 0 else float("nan")
    _ganancia = (1 - mae_bosque / mae_lineal) * 100

    mo.vstack([
        mo.md("### Comparación de modelos — exactitud *y* costo"),
        mo.ui.table(comparacion, selection=None, pagination=False),
        mo.md(
            f"El random forest reduce el error de {mae_lineal:.3f} a {mae_bosque:.3f} "
            f"unidades de magnitud, cerca de un {_ganancia:.0f}% menos. Esa ganancia es el "
            "valor de modelar la no linealidad, pagado con un modelo que ya no se puede "
            "leer como un conjunto de coeficientes.\\n\\n"
            f"Pero ahora véase la columna de costo: el bosque tardó **{_costo:,.0f}× más en "
            f"entrenarse** ({s_entrena_bosque:.3f} s contra {s_entrena_lineal * 1000:.1f} ms), "
            "y también es más lento al predecir, porque cada predicción debe recorrer 200 "
            "árboles en vez de evaluar un solo producto punto.\\n\\n"
            "**Este es el compromiso que importa en la práctica.** La exactitud no es "
            "gratis. Con esta cantidad de datos unos segundos son irrelevantes, pero la "
            "misma razón aplicada a una búsqueda de hiperparámetros con cientos de "
            "configuraciones, o a un modelo reentrenado cada hora, es la diferencia entre "
            "un trabajo de laptop y uno de clúster. Reportar exactitud sin reportar costo "
            "esconde la mitad de la decisión de ingeniería."
        ),
    ])
    return (comparacion,)


@app.cell
def _(VARIABLES, alt, mo, modelo_bosque, pl):
    importancias = (
        pl.DataFrame({
            "variable": VARIABLES,
            "importancia": modelo_bosque.feature_importances_,
        })
        .sort("importancia", descending=True)
    )

    grafico_importancia = (
        alt.Chart(importancias)
        .mark_bar(cornerRadiusEnd=3)
        .encode(
            x=alt.X("importancia:Q", title="Importancia relativa"),
            y=alt.Y("variable:N", title=None, sort="-x"),
            color=alt.Color("importancia:Q", scale=alt.Scale(
                scheme="viridis"), legend=None),
            tooltip=["variable:N", "importancia:Q"],
        )
        .properties(title="Importancia de variables del random forest", width=600, height=280)
    )

    mo.vstack([
        mo.md("#### En qué se apoya el bosque"),
        grafico_importancia,
        mo.md(
            "El orden difiere del que dio el modelo lineal, y esa diferencia es "
            "informativa: aquí la importancia mide cuánto reduce cada variable la impureza "
            "a lo largo de todas las divisiones, lo que premia a las variables con "
            "estructura *no lineal* útil aun cuando su correlación lineal sea débil."
        ),
    ])
    return


@app.cell
def _(alt, mo, pl, pred_bosque, pred_lineal, y_prueba):
    # Predicho contra real es el diagnóstico más informativo en regresión: un modelo
    # perfecto pone todos los puntos sobre la diagonal.
    predicciones = pl.concat([
        pl.DataFrame({
            "real": y_prueba,
            "predicho": pred_lineal,
            "modelo": ["Regresión lineal"] * len(y_prueba),
        }),
        pl.DataFrame({
            "real": y_prueba,
            "predicho": pred_bosque,
            "modelo": ["Random forest"] * len(y_prueba),
        }),
    ])

    # Altair no puede facetar un gráfico en capas salvo que los datos vivan en el nivel
    # de la faceta, así que los adjuntamos en `.facet(...)` y dejamos las capas sin datos.
    dispersion = (
        alt.Chart()
        .mark_circle(opacity=0.25, size=18)
        .encode(
            x=alt.X("real:Q", title="Magnitud real"),
            y=alt.Y("predicho:Q", title="Magnitud predicha"),
            color=alt.Color("modelo:N", legend=None),
            tooltip=["real:Q", "predicho:Q"],
        )
    )

    # Línea de referencia y = x, trazada a partir del propio rango de los datos.
    diagonal = (
        alt.Chart()
        .mark_line(color="black", strokeDash=[4, 4], opacity=0.6)
        .encode(
            x=alt.X("real:Q"),
            y=alt.Y("real:Q"),
        )
    )

    grafico_predicciones = (
        alt.layer(dispersion, diagonal, data=predicciones)
        .properties(width=300, height=300)
        .facet(column=alt.Column("modelo:N", title=None))
    )

    mo.vstack([
        mo.md("#### Predicho contra real — el diagnóstico que importa"),
        grafico_predicciones,
        mo.md(
            "La línea punteada es la predicción perfecta. La nube del modelo lineal es "
            "visiblemente más ancha y muestra una curvatura sistemática: sobreestima los "
            "eventos pequeños y subestima los grandes, la firma clásica de ajustar una "
            "recta a una relación no lineal. La nube del bosque se ciñe mucho más a la "
            "diagonal."
        ),
    ])
    return


@app.cell
def _(RandomForestRegressor, X, cross_val_score, mo, y):
    # Una sola partición puede salir afortunada. La validación cruzada reajusta el
    # modelo sobre varias particiones distintas y reporta la dispersión, que es la
    # forma honesta de enunciar un número de desempeño.
    bosque_vc = RandomForestRegressor(
        n_estimators=100, min_samples_leaf=2, random_state=42, n_jobs=-1
    )
    puntajes_vc = cross_val_score(
        bosque_vc, X, y, cv=5, scoring="r2", n_jobs=-1)

    mo.md(f"""
    ### Validación cruzada — ¿es estable el resultado?

    Validación cruzada de 5 particiones, R² en cada partición retenida:

    | Partición | R² |
    |---|---|
    {chr(10).join(f"| {i + 1} | {s:.3f} |" for i, s in enumerate(puntajes_vc))}
    | **Media ± desviación** | **{puntajes_vc.mean():.3f} ± {puntajes_vc.std():.3f}** |

    Una desviación estándar pequeña entre particiones significa que el modelo no depende
    de una partición favorable: el resultado es real, no un artefacto de cómo nos tocó
    dividir los datos. Reportar un único puntaje de prueba sin esta comprobación es una de
    las formas más fáciles de engañarse a uno mismo.

    Nótese otra vez `n_jobs=-1`: las cinco particiones son ajustes independientes, así que
    la validación cruzada es en sí misma vergonzosamente paralela. En una estación de
    trabajo esto es casi gratis; en un clúster es la manera de escalar búsquedas de
    hiperparámetros a miles de configuraciones.
    """)
    return (puntajes_vc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.2 — ¿Qué tan bien escala el bosque entre núcleos?

    Afirmamos que un random forest es "vergonzosamente paralelo". Esa es una afirmación
    sobre el hardware, así que merece una medición y no una aseveración. Abajo entrenamos
    el mismo bosque con `n_jobs` en 1, 2, 4, … hasta la cantidad de núcleos lógicos, y
    calculamos la **aceleración** y la **eficiencia paralela** (aceleración ÷ trabajadores).

    Es de esperar que la curva se aleje de la línea ideal. Tres razones, todas ellas
    contenido estándar de HPC:

    1. **Ley de Amdahl** — el remuestreo, la preparación y la agregación de resultados
       son partes seriales.
    2. **Los hyperthreads no son núcleos** — pasado el número de núcleos físicos, los
       trabajadores comparten unidades de ejecución y la ganancia se aplana.
    3. **Costo de arranque de los trabajadores** — el backend `loky` de joblib lanza
       procesos, y con esta cantidad de datos ese costo fijo es una fracción visible del
       total.
    """)
    return


@app.cell
def _(HARDWARE, RandomForestRegressor, X_entrena, mo, pl, time, y_entrena):
    # Barrido de n_jobs en potencias de dos hasta el número de núcleos lógicos.
    _max_trabajadores = HARDWARE["nucleos_logicos"] or 1
    cantidades_trabajadores, _t = [], 1
    while _t <= _max_trabajadores:
        cantidades_trabajadores.append(_t)
        _t *= 2
    if _max_trabajadores not in cantidades_trabajadores:
        cantidades_trabajadores.append(_max_trabajadores)

    filas_escalado = []
    for _n_jobs in cantidades_trabajadores:
        _modelo = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=_n_jobs
        )
        # Mejor de 2: suficiente para descartar una corrida desafortunada sin que el
        # barrido domine el tiempo de ejecución del cuaderno.
        _tiempos = []
        for _ in range(2):
            _t0 = time.perf_counter()
            _modelo.fit(X_entrena, y_entrena)
            _tiempos.append(time.perf_counter() - _t0)
        filas_escalado.append(
            {"n_jobs": _n_jobs, "s_entrenamiento": round(min(_tiempos), 4)})

    _serial = filas_escalado[0]["s_entrenamiento"]
    escalado = pl.DataFrame(filas_escalado).with_columns(
        (_serial / pl.col("s_entrenamiento")).round(2).alias("aceleracion"),
        ((_serial / pl.col("s_entrenamiento")) / pl.col("n_jobs") * 100)
        .round(0)
        .alias("eficiencia_pct"),
    )

    mo.vstack([
        mo.md(
            "### Tiempo de entrenamiento del random forest contra cantidad de trabajadores"),
        mo.ui.table(escalado, selection=None, pagination=False),
        mo.md(
            f"Núcleos físicos en este entorno: **{HARDWARE['nucleos_fisicos']}**; "
            f"núcleos lógicos: **{HARDWARE['nucleos_logicos']}**. La eficiencia es la "
            "aceleración dividida entre los trabajadores, así que 100% es escalado "
            "perfecto. Conviene observar dónde cae: ese codo suele estar en la cantidad de "
            "núcleos físicos, no en la de lógicos."
        ),
    ])
    return (escalado,)


@app.cell
def _(alt, escalado, pl):
    # Aceleración medida contra la línea ideal de escalado lineal.
    _ideal = pl.DataFrame({
        "n_jobs": escalado["n_jobs"],
        "aceleracion": escalado["n_jobs"].cast(pl.Float64),
        "serie": ["Ideal (lineal)"] * escalado.height,
    })
    _medida = escalado.select(
        "n_jobs",
        pl.col("aceleracion").cast(pl.Float64),
        pl.lit("Medida").alias("serie"),
    )

    grafico_escalado = (
        alt.Chart(pl.concat([_medida, _ideal]))
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("n_jobs:Q", title="n_jobs (trabajadores en paralelo)"),
            y=alt.Y("aceleracion:Q", title="Aceleración respecto a n_jobs=1"),
            color=alt.Color("serie:N", title=None),
            strokeDash=alt.StrokeDash("serie:N", legend=None),
            tooltip=["serie:N", "n_jobs:Q", "aceleracion:Q"],
        )
        .properties(title="Escalado paralelo del random forest", width=600, height=320)
        .configure_axis(labelFontSize=12, titleFontSize=14)
        .configure_title(fontSize=16)
    )

    grafico_escalado
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3.3 — Registro de los tiempos

    Todo lo que medimos se escribe a un CSV con la **misma convención que el resto de
    este curso**: líneas de metadata `# clave: valor` que describen el entorno que produjo
    los números, seguidas de la tabla.

    Esto importa más de lo que parece. Un tiempo sin su contexto de hardware no es un
    resultado: es un número. Registrar el entorno junto a la medición es lo que hace que
    la comparación sea reproducible, y es la razón por la que el archivo indica con
    claridad que se trata de **tiempos de CPU**.
    """)
    return


@app.cell
def _(HARDWARE, comparacion, escalado, mo, puntajes_vc):
    import csv
    import pathlib

    CSV_TIEMPOS = pathlib.Path("tiempos_05_polars_ml.csv")

    # Formato largo: una medición por línea, con la sección que la produjo.
    _filas = []
    for _r in comparacion.iter_rows(named=True):
        _filas.append(["entrenamiento_modelo", _r["modelo"],
                      "s_entrenamiento", _r["s_entrenamiento"]])
        _filas.append(["entrenamiento_modelo", _r["modelo"],
                      "s_prediccion", _r["s_prediccion"]])
        _filas.append(["calidad_modelo", _r["modelo"], "r2", _r["r2"]])
        _filas.append(["calidad_modelo", _r["modelo"], "mae", _r["mae"]])
    for _r in escalado.iter_rows(named=True):
        _filas.append(
            ["escalado_bosque", f"n_jobs={_r['n_jobs']}", "s_entrenamiento", _r["s_entrenamiento"]])
        _filas.append(
            ["escalado_bosque", f"n_jobs={_r['n_jobs']}", "aceleracion", _r["aceleracion"]])
    _filas.append(["validacion_cruzada", "Random forest (5 particiones)",
                  "r2_media", round(float(puntajes_vc.mean()), 4)])
    _filas.append(["validacion_cruzada", "Random forest (5 particiones)",
                  "r2_desviacion", round(float(puntajes_vc.std()), 4)])

    with open(CSV_TIEMPOS, "w", newline="") as _f:
        for _k, _v in HARDWARE.items():
            _f.write(f"# {_k}: {_v}\n")
        # Dejamos explícito el dispositivo de cómputo: scikit-learn corre en CPU.
        _f.write("# dispositivo: CPU (scikit-learn es una biblioteca de CPU)\n")
        _w = csv.writer(_f, lineterminator="\n")
        _w.writerow(["seccion", "etiqueta", "metrica", "valor"])
        _w.writerows(_filas)

    mo.md(f"""
    ### Tiempos guardados

    Se escribió **`{CSV_TIEMPOS}`** — {len(_filas)} mediciones, precedidas por
    {len(HARDWARE) + 1} líneas de metadata que describen este entorno.

    ```
    # entorno: {HARDWARE["entorno"]}
    # sklearn: {HARDWARE["sklearn"]}
    # nucleos_logicos: {HARDWARE["nucleos_logicos"]}
    # nucleos_fisicos: {HARDWARE["nucleos_fisicos"]}
    # dispositivo: CPU (scikit-learn es una biblioteca de CPU)
    seccion,etiqueta,metrica,valor
    ...
    ```

    En Colab se descarga desde la pestaña **Archivos** de la barra lateral izquierda. En
    el repositorio del curso este archivo va en `data/reference_timings/`, junto a los
    tiempos registrados de los demás cuadernos, para que quienes no tengan una máquina
    comparable igual cuenten con números que discutir.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Conclusiones

    ### Dataframes y análisis exploratorio
    - **polars** destaca al cargar y transformar archivos CSV grandes, con evaluación
      diferida y operaciones vectorizadas: habilidades críticas para flujos de HPC.
    - **La interfaz reactiva de marimo** (`mo.ui.slider`, `mo.ui.dropdown`, `mo.ui.table`)
      permite construir tableros interactivos con muy poco código repetitivo. Cada celda
      se vuelve a ejecutar sola cuando cambian sus entradas.
    - **SQL de DuckDB** (mediante `mo.sql`) se integra sin fricción con los dataframes de
      polars, y da lo mejor de ambos paradigmas de consulta.
    - **Altair** ofrece gráficos declarativos e interactivos que funcionan bien tanto con
      polars como con pandas.


    ### Hardware, costo y medición honesta
    - **Conocer los propios núcleos.** `n_jobs=-1` cuenta núcleos *lógicos*, pero la curva
      de escalado se dobla en la cantidad de núcleos *físicos*, porque los hyperthreads
      comparten unidades de ejecución.
    - **Cuidado con la sobresuscripción de hilos.** Los grupos de BLAS y OpenMP están por
      debajo de scikit-learn y se multiplican con `n_jobs`. `threadpoolctl` sirve para
      inspeccionarlos.
    - **Reportar el costo junto a la exactitud.** El bosque fue mucho más exacto *y*
      órdenes de magnitud más caro de entrenar. Una tabla de resultados que solo muestra
      exactitud esconde la mitad de la decisión de ingeniería.
    - **Un tiempo sin su hardware no es un resultado.** Cada número aquí se escribe a CSV
      junto con el entorno que lo produjo.
    - **La GPU se elige cambiando de biblioteca, no de parámetro.** scikit-learn es de
      CPU; RAPIDS (cuML, cuDF) y XGBoost son la vía a la GPU, y solo compensan cuando la
      razón entre cómputo y transferencia de datos es alta.

    ### Próximos pasos
    - Sustituir la dirección del conjunto de datos por el flujo `all_day.csv` para tener
      datos más pequeños y rápidos
    - Experimentar con `pl.scan_csv()` y `pl.LazyFrame` para procesamiento fuera de memoria
    - Volver a correr las mediciones con el control de escala al máximo y ver cómo se
      ensancha la diferencia
    - Agregar potenciación por gradiente (`HistGradientBoostingRegressor`) a la comparación
    - Ejecutar la Parte 4 en un entorno con GPU y comparar cuML contra scikit-learn con los
      datos propios
    - Probar a predecir `depth` en vez de `mag`, o a clasificar `type` (sismo contra
      voladura de cantera)
    - Explorar consultas espaciales (por ejemplo, recuadros delimitadores por región)
    - Exportar los resultados filtrados a Parquet para trabajos de HPC posteriores
    """)
    return


if __name__ == "__main__":
    app.run()
