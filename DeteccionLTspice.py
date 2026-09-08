"""
Clasificador por reglas simples (voz / golpe / ruido) usando la señal
FILTRADA que exportas directamente de LTspice.
"""

import sys
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# CONFIGURACIÓN Y UMBRALES -- esto es lo que vas a ir ajustando
# ---------------------------------------------------------------
ARCHIVO_LTSPICE = sys.argv[1] if len(sys.argv) > 1 else "AudioFiltrado3.txt"

FS_OBJETIVO = 16000   # Hz -- tasa a la que remuestreamos la señal para procesarla
FRAME_MS = 20         # tamaño de cada ventana de análisis (20 ms -- ANTES estaba en 1 ms, demasiado corto)
HOP_MS = 10           # cuánto se desliza la ventana entre cálculos (50% solape)

UMBRAL_ENERGIA_DB = -45          # piso mínimo absoluto para considerar "hay algo sonando"
DURACION_GOLPE_MAX_S = 0.08      # eventos de menos de 80 ms -> se consideran "golpe"
DURACION_GOLPE_MAX_S = 0.15     # golpes reales: 110-120ms; voz real: 350-770ms
RATIO_ALTA_RUIDO_MIN = 0.20  # ruido real: 21-33% en banda alta; voz/golpe: 2-12%
GAP_MERGE_S = 0.02    # bajado de 0.05 a 0.02 -- el valor anterior fusionaba ruido con voz real


def leer_export_ltspice(path):
    df = pd.read_csv(path, sep=None, engine="python", header=None)
    try:
        float(str(df.iloc[0, 0]).replace(",", "."))
    except ValueError:
        df = df.iloc[1:].reset_index(drop=True)
    t = df.iloc[:, 0].astype(str).str.replace(",", ".").astype(float).to_numpy()
    v = df.iloc[:, 1].astype(str).str.replace(",", ".").astype(float).to_numpy()
    return t, v


def remuestrear_uniforme(t, v, fs):
    t_uniforme = np.arange(t[0], t[-1], 1 / fs)
    v_uniforme = np.interp(t_uniforme, t, v)
    return v_uniforme, fs


def quitar_bias_dc(y):
    return y - np.mean(y)


def energia_por_ventana(y, sr):
    frame_len = int(sr * FRAME_MS / 1000)
    hop_len = int(sr * HOP_MS / 1000)
    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    tiempos = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_len)
    return rms_db, tiempos


def detectar_eventos(rms_db, tiempos, umbral_db):
    activo = rms_db > umbral_db
    eventos = []
    inicio = None
    for i, a in enumerate(activo):
        if a and inicio is None:
            inicio = tiempos[i]
        elif not a and inicio is not None:
            eventos.append((inicio, tiempos[i]))
            inicio = None
    if inicio is not None:
        eventos.append((inicio, tiempos[-1]))
    return eventos


def fusionar_eventos_cercanos(eventos, gap_max_s):
    """
    Si el silencio entre dos eventos detectados es muy corto (ej. < 50ms),
    probablemente es UN SOLO evento real que quedó partido por una caída
    momentánea de energía (una sílaba entre golpe y golpe de una misma
    ráfaga, o ruido en la medición). Los fusionamos en uno solo.
    """
    if not eventos:
        return eventos
    fusionados = [eventos[0]]
    for (t_ini, t_fin) in eventos[1:]:
        t_ini_prev, t_fin_prev = fusionados[-1]
        if t_ini - t_fin_prev < gap_max_s:
            fusionados[-1] = (t_ini_prev, t_fin)  # extiende el evento anterior
        else:
            fusionados.append((t_ini, t_fin))
    return fusionados


def zcr_evento(y, sr, t_ini, t_fin):
    i0, i1 = int(t_ini * sr), int(t_fin * sr)
    segmento = y[i0:i1]
    if len(segmento) < 2:
        return 0.0
    return float(np.mean(librosa.feature.zero_crossing_rate(segmento)))


def energia_por_banda(y, sr, t_ini, t_fin):
    i0, i1 = int(t_ini * sr), int(t_fin * sr)
    segmento = y[i0:i1]
    if len(segmento) < 32:
        return 0, 0, 0
    espectro = np.abs(np.fft.rfft(segmento))
    freqs = np.fft.rfftfreq(len(segmento), 1 / sr)
    banda_baja = espectro[(freqs >= 100) & (freqs < 500)].sum()
    banda_media = espectro[(freqs >= 500) & (freqs < 1500)].sum()
    banda_alta = espectro[(freqs >= 1500) & (freqs < 3500)].sum()
    total = banda_baja + banda_media + banda_alta + 1e-10
    return banda_baja / total , banda_media / total, banda_alta / total


def clasificar_evento(duracion_s, ratio_alta):
    # ruido: mucha energía en alta frecuencia (banda ancha), sin importar duración
    if ratio_alta > RATIO_ALTA_RUIDO_MIN:
        return "RUIDO"
    # entre lo que no es ruido: corto = golpe, largo = voz
    if duracion_s < DURACION_GOLPE_MAX_S:
        return "GOLPE"
    return "VOZ"


def main():
    print(f"Leyendo {ARCHIVO_LTSPICE} ...")
    t, v = leer_export_ltspice(ARCHIVO_LTSPICE)
    print(f"  {len(t)} muestras originales, duración {t[-1]-t[0]:.3f} s")

    y, sr = remuestrear_uniforme(t, v, FS_OBJETIVO)
    y = quitar_bias_dc(y)
    print(f"  Remuestreado a {sr} Hz uniforme, {len(y)} muestras\n")

    rms_db, tiempos = energia_por_ventana(y, sr)
    piso_ruido_db = np.percentile(rms_db, 20)
    umbral = max(UMBRAL_ENERGIA_DB, piso_ruido_db + 6)
    print(f"Piso de ruido estimado: {piso_ruido_db:.1f} dB  ->  umbral: {umbral:.1f} dB\n")

    eventos_crudos = detectar_eventos(rms_db, tiempos, umbral)
    eventos = fusionar_eventos_cercanos(eventos_crudos, GAP_MERGE_S)
    print(f"Eventos crudos: {len(eventos_crudos)}  ->  tras fusionar cercanos: {len(eventos)}\n")

    resultados = []
    for (t_ini, t_fin) in eventos:
        dur = t_fin - t_ini
        if dur < 0.02:
            continue
        zcr = zcr_evento(y, sr, t_ini, t_fin)
        b_baja, b_media, b_alta = energia_por_banda(y, sr, t_ini, t_fin)
        clase = clasificar_evento(dur, b_alta)
        resultados.append((t_ini, t_fin, clase))
        print(f"[{t_ini:5.2f}s - {t_fin:5.2f}s]  dur={dur*1000:5.0f}ms  "
              f"ZCR={zcr:.3f}  banda(baja/media/alta)=({b_baja:.2f}/{b_media:.2f}/{b_alta:.2f})  "
              f"-> {clase}")

    plt.figure(figsize=(14, 4))
    plt.plot(tiempos, rms_db, label="Energía (dB)", linewidth=0.8)
    plt.axhline(umbral, color="orange", linestyle="--", label="Umbral de detección")
    colores = {"VOZ": "green", "GOLPE": "red", "RUIDO": "gray"}
    for idx, (t_ini, t_fin, clase) in enumerate(resultados):
        plt.axvspan(t_ini, t_fin, color=colores[clase], alpha=0.3)
        y_texto = max(rms_db) - (idx % 3) * 8  # alterna altura de texto para que no se amontone
        plt.text((t_ini + t_fin) / 2, y_texto, clase, ha="center", fontsize=8)
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Energía (dB)")
    plt.title(f"Clasificación por reglas -- señal de LTspice ({ARCHIVO_LTSPICE})")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("clasificacion_ltspice.png", dpi=120)
    print("\nGráfica guardada en clasificacion_ltspice.png")


if __name__ == "__main__":
    main()