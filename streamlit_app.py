import hashlib
import io
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Compresión de imágenes con SVD", layout="wide")

# ---------------------------------------------------------------
# DATOS DE ENTRADA
# ---------------------------------------------------------------
K_INICIAL = 50          # k con el que arranca la barra
K_MAX_ALEATORIO = 200   # valores singulares que calcula el SVD aleatorio
SOBREMUESTREO = 10      # columnas extra del SVD aleatorio (mejoran la precision)
ITERACIONES = 2         # iteraciones de potencia del SVD aleatorio
SEMILLA = 42            # semilla fija: el SVD aleatorio se repite igual
BYTES_VALOR = 4         # cada valor guardado ocupa 4 bytes (float32)
CANALES = 3
MAX_MEGAPIXELES = 6     # imagenes mas grandes se rechazan (protegen la memoria del servidor)


# ---------------------------------------------------------------
# FUNCIONES
# ---------------------------------------------------------------
def leer_imagen(datos, extension):
    """Lee la imagen desde sus bytes y la devuelve como matriz (alto, ancho, 3) con valores entre 0 y 1."""
    image = plt.imread(io.BytesIO(datos), format=extension)
    if image.dtype == np.uint8:
        image = image / 255.0
    if image.ndim == 2:
        image = np.stack((image, image, image), axis=2)
    return image[:, :, :3].astype(float)


def svd_clasico(A):
    return np.linalg.svd(A, full_matrices=False)


def svd_aleatorio(A, k_max, sobremuestreo, iteraciones, semilla):
    """SVD aleatorio: calcula solo los k_max valores singulares mas grandes."""
    rng = np.random.default_rng(semilla)
    m, n = A.shape
    l = min(k_max + sobremuestreo, min(m, n))

    # 1. proyeccion aleatoria: A * Omega captura el rango principal de A
    Omega = rng.standard_normal((n, l))
    Q, _ = np.linalg.qr(A @ Omega)

    # 2. iteraciones de potencia (con QR en cada paso para mantener la ortogonalidad)
    for i in range(iteraciones):
        Z, _ = np.linalg.qr(A.T @ Q)
        Q, _ = np.linalg.qr(A @ Z)

    # 3. SVD de la matriz pequeña B = Q^T * A y regreso al espacio original
    B = Q.T @ A
    Ub, s, Vt = np.linalg.svd(B, full_matrices=False)
    U = Q @ Ub

    return U[:, :k_max], s[:k_max], Vt[:k_max, :]


def calcular_svd(image, metodo):
    """Un SVD por canal (R, G, B). Devuelve la lista de 3 tuplas (U, s, Vt) y el tiempo."""
    tiempo_inicio = time.time()
    if metodo == "clasico":
        svds = [svd_clasico(image[:, :, c]) for c in range(CANALES)]
    else:
        k_max = min(K_MAX_ALEATORIO, min(image.shape[0], image.shape[1]))
        svds = [svd_aleatorio(image[:, :, c], k_max, SOBREMUESTREO, ITERACIONES, SEMILLA) for c in range(CANALES)]
    return svds, time.time() - tiempo_inicio


# Versiones con memoria (cache): el SVD se calcula una sola vez por imagen y metodo.
# Los parametros que empiezan con "_" no se usan para identificar la imagen (se usa la huella).
@st.cache_resource(show_spinner=False, max_entries=4)
def cargar_imagen(huella, _datos, extension):
    return leer_imagen(_datos, extension)


@st.cache_resource(show_spinner=False, max_entries=8)
def calcular_svd_cache(huella, _datos, extension, metodo):
    return calcular_svd(cargar_imagen(huella, _datos, extension), metodo)


def reconstruir_color(svds, k):
    """Reconstruye la imagen a color con los primeros k valores singulares de cada canal."""
    canales_rec = [U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :] for (U, s, Vt) in svds]
    return np.clip(np.stack(canales_rec, axis=2), 0, 1)


def energia_acumulada(s, energia_total):
    """Energia conservada para k = 1, 2, ... (usa los cuadrados de los valores singulares)."""
    return np.cumsum(s**2) / energia_total


def calcular_curvas(svds, image):
    """Energia acumulada (promedio de los 3 canales) y error relativo teorico para cada k."""
    energias = [np.sum(image[:, :, c]**2) for c in range(CANALES)]
    E_promedio = sum([energia_acumulada(svds[c][1], energias[c]) for c in range(CANALES)]) / CANALES
    total = sum(energias)
    descartada = total - sum([np.cumsum(svds[c][1]**2) for c in range(CANALES)])
    error_teorico = np.sqrt(np.maximum(descartada, 0) / total)
    return E_promedio, error_teorico


def calcular_error(original, reconstruida):
    """Error relativo de Frobenius."""
    return np.sqrt(np.sum((original - reconstruida)**2) / np.sum(original**2))


def k_recomendado(error_teorico, umbral):
    """Menor k cuyo error no supera el umbral (None si no se alcanza)."""
    cumplen = np.where(error_teorico <= umbral)[0]
    if len(cumplen) == 0:
        return None
    return int(cumplen[0]) + 1


def peso_npz_kb(svds, k):
    """Peso real (en KB) de guardar las matrices truncadas en un archivo .npz."""
    datos = {}
    for nombre, (U, s, Vt) in zip("RGB", svds):
        datos["U" + nombre] = U[:, :k].astype(np.float32)
        datos["s" + nombre] = s[:k].astype(np.float32)
        datos["Vt" + nombre] = Vt[:k, :].astype(np.float32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **datos)
    return buffer.getbuffer().nbytes / 1024


def segundos_de_carga(peso_kb, velocidad_mbps):
    return peso_kb * 1024 * 8 / (velocidad_mbps * 1e6)


def a_uint8(imagen):
    return (np.clip(imagen, 0, 1) * 255).round().astype(np.uint8)


def extension_de(nombre):
    return os.path.splitext(nombre)[1].lower().lstrip(".")


def ir_a_recomendado(clave, valor):
    st.session_state[clave] = valor


ENCABEZADOS = ["Imagen", "Tamaño (px)", "Peso original (KB)", "k recomendado", "k usado (aleatorio)",
               "Peso SVD .npz (KB)", "SVD / original", "Error clásico (%)", "Error aleatorio (%)",
               "Tiempo clásico (s)", "Tiempo aleatorio (s)", "Aleatorio más rápido (veces)",
               "Carga original (s)", "Carga SVD (s)"]


def calcular_fila(nombre, datos, umbral, velocidad):
    """Una fila de la tabla comparativa: los dos metodos sobre una imagen."""
    img = leer_imagen(datos, extension_de(nombre))
    if img.shape[0] * img.shape[1] > MAX_MEGAPIXELES * 1e6:
        raise ValueError("imagen demasiado grande")
    peso = len(datos) / 1024
    svd_c, t_c = calcular_svd(img, "clasico")
    svd_a, t_a = calcular_svd(img, "aleatorio")

    k_rec = k_recomendado(calcular_curvas(svd_c, img)[1], umbral)
    k_a = min(k_rec, len(svd_a[0][1]))
    err_c = calcular_error(img, reconstruir_color(svd_c, k_rec))
    err_a = calcular_error(img, reconstruir_color(svd_a, k_a))
    peso_svd = peso_npz_kb(svd_c, k_rec)

    return [nombre, str(img.shape[1]) + "x" + str(img.shape[0]), round(peso, 1), k_rec, k_a,
            round(peso_svd, 1), round(peso_svd / peso, 2), round(err_c * 100, 2), round(err_a * 100, 2),
            round(t_c, 2), round(t_a, 2), round(t_c / t_a, 1),
            round(segundos_de_carga(peso, velocidad), 2), round(segundos_de_carga(peso_svd, velocidad), 2)]


# ---------------------------------------------------------------
# BARRA LATERAL
# ---------------------------------------------------------------
st.sidebar.title("Compresión de imágenes con SVD")
st.sidebar.caption("Caso de estudio: imágenes de productos de Falabella")

subidas = st.sidebar.file_uploader("Sube tus imágenes (JPG, PNG o WebP)", type=["jpg", "jpeg", "png", "webp"],
                                   accept_multiple_files=True)

if len(subidas) == 0:
    st.title("Compresión de imágenes con SVD")
    st.info("Sube una o más imágenes desde la barra lateral para empezar. "
            "Las imágenes no se guardan: solo se usan mientras la página está abierta.")
    st.stop()

indice = st.sidebar.selectbox("Imagen", range(len(subidas)), format_func=lambda i: subidas[i].name)
metodo_nombre = st.sidebar.radio("Método de SVD", ["SVD clásico", "SVD aleatorio"])
umbral_error = st.sidebar.slider("Error máximo aceptado (%)", 1, 20, 5) / 100
velocidad_mbps = st.sidebar.slider("Velocidad de conexión supuesta (Mbps)", 1, 100, 5)


# ---------------------------------------------------------------
# CARGA Y SVD (se calculan una sola vez por imagen)
# ---------------------------------------------------------------
archivo = subidas[indice].name
datos = subidas[indice].getvalue()
extension = extension_de(archivo)
huella = hashlib.md5(datos).hexdigest()
peso_archivo_kb = len(datos) / 1024

try:
    image = cargar_imagen(huella, datos, extension)
except Exception:
    st.error("No se pudo leer la imagen " + archivo + ". Prueba con otro archivo JPG, PNG o WebP.")
    st.stop()
filas, columnas, _ = image.shape

if filas * columnas > MAX_MEGAPIXELES * 1e6:
    st.error("La imagen " + archivo + " tiene " + f"{filas * columnas / 1e6:.1f}" + " megapíxeles. "
             "Para no saturar el servidor, esta app acepta hasta " + str(MAX_MEGAPIXELES) + " megapíxeles.")
    st.stop()

with st.spinner("Calculando el SVD de la imagen (solo la primera vez)..."):
    svds_clasico, tiempo_clasico = calcular_svd_cache(huella, datos, extension, "clasico")
    svds_aleatorio, tiempo_aleatorio = calcular_svd_cache(huella, datos, extension, "aleatorio")

if metodo_nombre == "SVD clásico":
    svds, tiempo_svd = svds_clasico, tiempo_clasico
else:
    svds, tiempo_svd = svds_aleatorio, tiempo_aleatorio

n = len(svds[0][1])     # k maximo disponible con el metodo elegido

E_promedio, error_teorico = calcular_curvas(svds, image)
k_rec_exacto = k_recomendado(calcular_curvas(svds_clasico, image)[1], umbral_error)
k_rec = min(k_rec_exacto, n)

k_igual_peso = int(peso_archivo_kb * 1024 / (BYTES_VALOR * CANALES * (filas + columnas + 1)))


# ---------------------------------------------------------------
# IMAGENES Y BARRA
# ---------------------------------------------------------------
st.title("Compresión de imágenes con SVD")
st.caption(archivo + "  |  " + str(columnas) + " x " + str(filas) + " px  |  " + metodo_nombre)

zona_imagenes = st.container()

clave_k = "k_" + huella + "_" + metodo_nombre
clave_ultimo = "k_ultimo_" + huella
cambio_metodo = st.session_state.get("metodo_previo") != metodo_nombre
st.session_state["metodo_previo"] = metodo_nombre
if clave_k not in st.session_state or cambio_metodo:
    # al cambiar de metodo se conserva el k que se estaba usando (sin pasar del maximo del metodo)
    st.session_state[clave_k] = min(st.session_state.get(clave_ultimo, K_INICIAL), n)

col_barra, col_boton = st.columns([5, 1], vertical_alignment="bottom")
with col_barra:
    k = st.slider("k (cantidad de valores singulares que se conservan)", 1, n, key=clave_k)
with col_boton:
    st.button("Ir a k recomendado", on_click=ir_a_recomendado, args=(clave_k, k_rec), width="stretch")

st.session_state[clave_ultimo] = k
imagen_reconstruida = reconstruir_color(svds, k)

with zona_imagenes:
    col_original, col_reconstruida = st.columns(2)
    col_original.image(a_uint8(image), caption="Original", width="stretch")
    col_reconstruida.image(a_uint8(imagen_reconstruida), caption="Reconstruida (k = " + str(k) + ")", width="stretch")


# ---------------------------------------------------------------
# DATOS DE COMPRESION Y CALIDAD
# ---------------------------------------------------------------
pixeles = filas * columnas
originales = pixeles * CANALES
comprimidos = k * (filas + columnas + 1) * CANALES
razon = originales / comprimidos
peso_svd_kb = comprimidos * BYTES_VALOR / 1024

energia = E_promedio[k - 1]
error = calcular_error(image, imagen_reconstruida)

fila1 = st.columns(4)
fila1[0].metric("Tamaño original", f"{originales:,} valores")
fila1[1].metric("Tamaño comprimido", f"{comprimidos:,} valores")
fila1[2].metric("Razón de compresión", f"{razon:.2f}")
fila1[3].metric("Tiempo del SVD", f"{tiempo_svd:.2f} s")

fila2 = st.columns(4)
fila2[0].metric("Energía conservada", f"{energia*100:.2f} %")
fila2[1].metric("Error relativo", f"{error*100:.2f} %")
fila2[2].metric("Peso del archivo / del SVD", f"{peso_archivo_kb:.1f} / {peso_svd_kb:.1f} KB")
fila2[3].metric("Carga estimada (archivo / SVD)",
                f"{segundos_de_carga(peso_archivo_kb, velocidad_mbps):.2f} / {segundos_de_carga(peso_svd_kb, velocidad_mbps):.2f} s")

if k_rec_exacto > n:
    st.warning("Con el SVD aleatorio (k máximo = " + str(n) + ") no se alcanza el error de " + f"{umbral_error*100:.0f}"
               + " %. El k recomendado exacto es " + str(k_rec_exacto) + " (SVD clásico).")
else:
    st.info("k recomendado: " + str(k_rec) + " (error <= " + f"{umbral_error*100:.0f}" + " %).  "
            + ("El SVD pesa menos que el archivo solo si k <= " + str(k_igual_peso) + "." if k_igual_peso >= 1
               else "Ningún k hace que el SVD pese menos que el archivo original."))
if razon < 1:
    st.warning("Con este k el SVD guarda más valores que la imagen original.")
if peso_svd_kb > peso_archivo_kb:
    st.warning("Con este k el SVD pesa más que el archivo original.")


# ---------------------------------------------------------------
# PESTAÑAS
# ---------------------------------------------------------------
tab_graficos, tab_comparar, tab_tabla = st.tabs(["Gráficos", "Clásico vs aleatorio", "Tabla de todas las imágenes"])

with tab_graficos:
    col_a, col_b = st.columns(2)
    ks = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ks, svds[0][1], color="red", linewidth=1, label="R")
    ax.plot(ks, svds[1][1], color="green", linewidth=1, label="G")
    ax.plot(ks, svds[2][1], color="blue", linewidth=1, label="B")
    ax.axvline(k, color="black", linestyle=":", linewidth=1.5)
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("valor singular")
    ax.set_title("Valores singulares (escala log)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    col_a.pyplot(fig, width="stretch")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(ks, E_promedio, "k--", linewidth=1)
    ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(k, color="red", linestyle=":", linewidth=1.5)
    ax.plot([k], [energia], "ro")
    ax.set_xlabel("k")
    ax.set_ylabel("E")
    ax.set_title("Energía acumulada")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    col_b.pyplot(fig, width="stretch")
    plt.close(fig)

with tab_comparar:
    n_clasico = len(svds_clasico[0][1])
    n_aleatorio = len(svds_aleatorio[0][1])
    k_a = min(k, n_aleatorio)
    err_c = calcular_error(image, reconstruir_color(svds_clasico, k))
    err_a = calcular_error(image, reconstruir_color(svds_aleatorio, k_a))
    primeros = min(n_aleatorio, 100)
    dif_sigma = max([np.max(np.abs(svds_clasico[c][1][:primeros] - svds_aleatorio[c][1][:primeros]) / svds_clasico[c][1][:primeros])
                     for c in range(CANALES)])

    c1, c2, c3 = st.columns(3)
    c1.metric("Tiempo SVD clásico", f"{tiempo_clasico:.2f} s")
    c2.metric("Tiempo SVD aleatorio", f"{tiempo_aleatorio:.2f} s")
    c3.metric("Aleatorio más rápido", f"{tiempo_clasico / tiempo_aleatorio:.1f} veces")

    c4, c5, c6 = st.columns(3)
    c4.metric("Error con k = " + str(k) + " (clásico)", f"{err_c*100:.3f} %")
    c5.metric("Error con k = " + str(k_a) + " (aleatorio)", f"{err_a*100:.3f} %")
    c6.metric("Diferencia máx. en los " + str(primeros) + " primeros σ", f"{dif_sigma*100:.3f} %")

    if k > n_aleatorio:
        st.caption("El SVD aleatorio solo calcula hasta k = " + str(n_aleatorio) + ", por eso se compara con ese valor.")

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(np.arange(1, n_clasico + 1), svds_clasico[0][1], color="tab:blue", linewidth=2, label="Clásico")
    ax.plot(np.arange(1, n_aleatorio + 1), svds_aleatorio[0][1], color="tab:orange", linestyle="--", linewidth=1.5, label="Aleatorio")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("valor singular")
    ax.set_title("Valores singulares del canal R: clásico contra aleatorio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, width="stretch")
    plt.close(fig)

with tab_tabla:
    st.write("Calcula los dos métodos sobre todas las imágenes subidas, con el *k* recomendado del SVD clásico. "
             "Puede tardar si hay imágenes grandes.")
    if st.button("Calcular tabla de todas las imágenes"):
        filas_tabla = []
        omitidas = []
        barra = st.progress(0.0, text="Procesando imágenes...")
        for i, subida in enumerate(subidas):
            barra.progress(i / len(subidas), text="Procesando " + subida.name + " (" + str(i + 1) + " de " + str(len(subidas)) + ")")
            try:
                filas_tabla.append(calcular_fila(subida.name, subida.getvalue(), umbral_error, velocidad_mbps))
            except Exception:
                omitidas.append(subida.name)
        barra.empty()
        if len(omitidas) > 0:
            st.warning("Se omitieron (no se pudieron leer o superan los " + str(MAX_MEGAPIXELES) + " megapíxeles): " + ", ".join(omitidas))
        st.session_state["tabla"] = (pd.DataFrame(filas_tabla, columns=ENCABEZADOS), umbral_error, velocidad_mbps)

    if "tabla" in st.session_state:
        df, umbral_tabla, velocidad_tabla = st.session_state["tabla"]
        st.caption("Tabla calculada con error máximo de " + f"{umbral_tabla*100:.0f}" + " % y conexión de "
                   + str(velocidad_tabla) + " Mbps. Si cambias esos valores o las imágenes, vuelve a calcularla.")
        st.dataframe(df, width="stretch", hide_index=True)
        csv = df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
        st.download_button("Descargar CSV", data=csv, file_name="resultados_svd.csv", mime="text/csv")
