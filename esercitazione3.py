import soundfile as sdf
import numpy as np
from matplotlib import pyplot as ppt
import scipy as scp
from scipy.signal import find_peaks, peak_widths

def analisi_audio(file_audio, nome_onda):

    data, samplerate = sdf.read(file_audio)

    print("Frequenza di campionamento:", samplerate)
    print("Shape dei dati:", data.shape)

    if data.ndim > 1:
        canale = data[:, 0]
    else:
        canale = data

    # Asse del tempo
    tempo = np.arange(len(canale)) / samplerate

    # FFT
    Y = scp.fft.fft(canale)

    # Frequenze associate ai coefficienti FFT
    freq = scp.fft.fftfreq(len(canale), d=1/samplerate)

    # Shift per ordinare bene lo spettro
    Y_shift = scp.fft.fftshift(Y)
    freq_shift = scp.fft.fftshift(freq)

    # Quantità richieste
    spettro_potenza = np.abs(Y_shift) ** 2
    parte_reale = np.real(Y_shift)
    parte_immaginaria = np.imag(Y_shift)

    # =========================
    # FIGURA 1: waveform + potenza
    # =========================
    fig1, axs1 = ppt.subplots(2, 1, figsize=(12, 10))
    fig1.suptitle(f'{nome_onda} - Waveform e Spettro di Potenza', fontsize=16)

    axs1[0].plot(tempo, canale)
    axs1[0].set_title(f'Waveform di {nome_onda}')
    axs1[0].set_xlabel("Tempo [s]")
    axs1[0].set_ylabel("Ampiezza")
    axs1[0].grid(True)

    axs1[1].plot(freq_shift, spettro_potenza)
    axs1[1].set_xlim(-3000, 3000)
    axs1[1].set_title(f'Spettro di potenza di {nome_onda}')
    axs1[1].set_xlabel("Frequenza [Hz]")
    axs1[1].set_ylabel("Potenza")
    axs1[1].grid(True)

    fig1.subplots_adjust(hspace=0.42, top=0.88)

    ppt.show()

    # =========================
    # FIGURA 2: reale + immaginaria
    # =========================
    fig2, axs2 = ppt.subplots(2, 1, figsize=(12, 10))
    fig2.suptitle(f'{nome_onda} - Parte Reale e Immaginaria', fontsize=16)

    axs2[0].plot(freq_shift, parte_reale)
    axs2[0].set_xlim(-3000, 3000)
    axs2[0].set_title(f'Parte reale dei coefficienti di {nome_onda}')
    axs2[0].set_xlabel("Frequenza [Hz]")
    axs2[0].set_ylabel("Parte reale")
    axs2[0].grid(True)

    axs2[1].plot(freq_shift, parte_immaginaria)
    axs2[1].set_xlim(-3000, 3000)
    axs2[1].set_title(f'Parte immaginaria dei coefficienti di {nome_onda}')
    axs2[1].set_xlabel("Frequenza [Hz]")
    axs2[1].set_ylabel("Parte immaginaria")
    axs2[1].grid(True)

    fig2.subplots_adjust(hspace=0.42, top=0.88)

    ppt.show()

    return canale, samplerate, Y_shift, freq_shift, spettro_potenza


file_audio1 = "pulita_semplice.wav"
pul_semp_canale, pul_semp_samplerate, pul_semp_Y_shift, freq_pul_semp, pul_semp_spettro = analisi_audio(file_audio1, "Pulita Semplice")

file_audio2 = "diapason.wav"
diapason_canale, diapason_samplerate, diapason_Y_shift, freq_diapason, diapason_spettro = analisi_audio(file_audio2, "Diapason")

file_audio3 = "distorta.wav"
distorta_canale, distorta_samplerate, distorta_Y_shift, freq_distorta, distorta_spettro = analisi_audio(file_audio3, "Distorta")


# Questa funzione converte una frequenza in Hz nella nota musicale più vicina.
# Si usa come riferimento LA4 = 440 Hz, che nel sistema MIDI corrisponde al numero 69.
#
# La formula:
# midi = 69 + 12 * log2(f / 440)
#
# calcola quanti semitoni la frequenza f si trova sopra o sotto il LA4.
# Si usa log2 perché salire di un'ottava significa raddoppiare la frequenza,
# e ogni ottava contiene 12 semitoni.
#
# Il round serve perché la frequenza misurata dalla FFT può non coincidere
# perfettamente con la frequenza teorica della nota.
#
# Dopo aver trovato il numero MIDI:
# - midi % 12 permette di trovare il nome della nota, perché le note si ripetono ogni 12 semitoni
# - midi // 12 - 1 permette di trovare l'ottava
#
# Infine calcolo anche la frequenza teorica della nota trovata, così posso confrontarla
# con la frequenza misurata nello spettro.

def frequenza_a_nota(f):
    note = ['DO', 'DO#', 'RE', 'RE#', 'MI', 'FA',
            'FA#', 'SOL', 'SOL#', 'LA', 'LA#', 'SI']

    if f <= 0:
        return "N/A"

    # LA4 = 440 Hz corrisponde al numero MIDI 69
    midi = round(69 + 12 * np.log2(f / 440.0))

    nome_nota = note[midi % 12]
    ottava = midi // 12 - 1

    freq_teorica = 440.0 * 2 ** ((midi - 69) / 12)

    return f"{nome_nota}{ottava} ({freq_teorica:.2f} Hz teorici)"
 


def analisi_traformata(spettro_potenza, frequenze, nome_onda, soglia=0.05): 
 #La soglia serve per settare l'altezza minima dello spettro di potenza, ovvero almeno il 5% rispetto al picco massimo.
 #Va considerato che dopo divido tutti gli spettri di potenza per quello massimo per normalizzare cosi da poter usare 0.05, senno' dovrei calcolare il 5% del picco massimo
 
 # Prendo solo frequenze positive sopra 1 Hz per escludere le negative dall'analisi
 maschera= frequenze>20 #Array di true o false

 freq_pos= frequenze[maschera]
 potenze_pos=spettro_potenza[maschera]

 potenze_normalizzate=  potenze_pos/max(potenze_pos)

 picchi, proprieta= find_peaks (
  potenze_normalizzate,
  height=soglia, #Sotto della soglia non prendo la frequenza, serve per escludere le non significative
  distance=20 #Vuol dire che trovato un dato valido gaurdo tot posizioni dell' array dopo, nello spettro un singolo picco reale può non essere una punta perfetta: può essere largo, frastagliato o avere piccole ondulazioni sopra.
 )
 
     # Calcolo la larghezza dei picchi a metà altezza, convenzionalmente si usa meta', come nela foto nella cartella
 larghezze = peak_widths(
  potenze_normalizzate,
  picchi,
  rel_height=0.5
 )

 #Larghezze è un arrary di array, dove in larghezze[0], sono contenuti gli scarti tra indici destra e sinistra del picco tagliato al 50% , in larghezza[2] c'è la posizione a sinistra e in [3] quella a destra

 df = freq_pos[1] - freq_pos[0] #Distanza in frequenze tra una frequenza campionata e un' altra, tanto la distanza è sempre quella
 larghezze_hz = larghezze[0] * df #Moltiplicando distanza in indici del widths e distanza tra ogni indice in frequenze, ottengo la distanza widths in frequenze
 
 # Ordino i picchi dal più potente al meno potente
 ordine = np.argsort(potenze_normalizzate[picchi])[::-1]

 picchi_ordinati = picchi[ordine]
 larghezze_ordinate = larghezze_hz[ordine]

 print(f"\nAnalisi picchi di {nome_onda}")
 print("--------------------------------")

 # Picco principale
 p = picchi_ordinati[0]

 f0 = freq_pos[p]
 larghezza_f0 = larghezze_ordinate[0]

 print("Picco principale:")
 print(f"Frequenza: {freq_pos[p]:.2f} Hz")
 print(f"Nota corrispondente: {frequenza_a_nota(freq_pos[p])}")
 print(f"Potenza relativa: {potenze_normalizzate[p]:.3f}")
 print(f"Potenza: {potenze_pos[p]:.2f}")
 print(f"Larghezza: {larghezze_ordinate[0]:.2f} Hz")

 # Primo picco secondario
 if len(picchi_ordinati) > 1:
    p = picchi_ordinati[1]

    print("\nPrimo picco secondario:")
    print(f"Frequenza: {freq_pos[p]:.2f} Hz")
    print(f"Nota corrispondente: {frequenza_a_nota(freq_pos[p])}")
    print(f"Potenza relativa: {potenze_normalizzate[p]:.3f}")
    print(f"Potenza: {potenze_pos[p]:.2f}")
    print(f"Larghezza: {larghezze_ordinate[1]:.2f} Hz")
 else:
    print("\nNon è stato trovato il primo picco secondario sopra soglia.")

 # Secondo picco secondario
 if len(picchi_ordinati) > 2:
    p = picchi_ordinati[2]

    print("\nSecondo picco secondario:")
    print(f"Frequenza: {freq_pos[p]:.2f} Hz")
    print(f"Nota corrispondente: {frequenza_a_nota(freq_pos[p])}")
    print(f"Potenza relativa: {potenze_normalizzate[p]:.3f}")
    print(f"Potenza: {potenze_pos[p]:.2f}")
    print(f"Larghezza: {larghezze_ordinate[2]:.2f} Hz")
 else:
    print("\nNon è stato trovato il secondo picco secondario sopra soglia.")

 return f0, larghezza_f0


f0_pul_semp, larg_pul_semp = analisi_traformata(pul_semp_spettro, freq_pul_semp, "Pulita Semplice")

f0_diapason, larg_diapason = analisi_traformata(diapason_spettro, freq_diapason, "Diapason")

f0_distorta, larg_distorta = analisi_traformata(distorta_spettro, freq_distorta, "Distorta")


def filtra_picco_principale(Y_shift, frequenze, samplerate, f0, nome_onda, nome_file_output, banda=30):

    # Filtro passa-banda centrato sul picco principale.
    # Tengo solo le frequenze comprese tra f0 - banda e f0 + banda.
    # Uso abs(frequenze) per tenere sia +f0 sia -f0,
    # perché il segnale reale ha spettro simmetrico.

    maschera = (
        (np.abs(frequenze) >= f0 - banda) &
        (np.abs(frequenze) <= f0 + banda)
    )

    # Creo una FFT filtrata tutta a zero
    Y_filtrato_shift = np.zeros_like(Y_shift)

    # Copio solo i coefficienti dentro la banda del picco principale
    Y_filtrato_shift[maschera] = Y_shift[maschera]

    # Calcolo lo spettro filtrato, utile se vuoi graficarlo
    spettro_filtrato = np.abs(Y_filtrato_shift) ** 2

    # Tolgo lo shift prima della IFFT
    Y_filtrato = scp.fft.ifftshift(Y_filtrato_shift)

    # Torno nel dominio del tempo
    segnale_filtrato = np.real(scp.fft.ifft(Y_filtrato))

    # Normalizzo per evitare clippingPer sicurezza. Nei file audio, valori troppo vicini o superiori a ±1 possono causare clipping, cioè il suono viene tagliato e distorto.
    massimo = np.max(np.abs(segnale_filtrato))
    if massimo > 0:
        segnale_filtrato = segnale_filtrato / massimo * 0.9

    # Salvo il nuovo file audio
    sdf.write(nome_file_output, segnale_filtrato, samplerate)

    print(f"\nFiltro applicato a {nome_onda}")
    print("--------------------------------")
    print("Tipo filtro: passa-banda")
    print(f"Picco principale mantenuto: {f0:.2f} Hz")
    print(f"Banda mantenuta: {f0 - banda:.2f} Hz - {f0 + banda:.2f} Hz")
    print(f"File audio creato: {nome_file_output}")

    return segnale_filtrato, Y_filtrato_shift, spettro_filtrato


pul_semp_filtrato, pul_semp_Y_filtrato, pul_semp_spettro_filtrato = filtra_picco_principale(
    pul_semp_Y_shift,
    freq_pul_semp,
    pul_semp_samplerate,
    f0_pul_semp,
    "Pulita Semplice",
    "pulita_semplice_filtrata.wav",
    banda=30
)

diapason_filtrato, diapason_Y_filtrato, diapason_spettro_filtrato = filtra_picco_principale(
    diapason_Y_shift,
    freq_diapason,
    diapason_samplerate,
    f0_diapason,
    "Diapason",
    "diapason_filtrato.wav",
    banda=30
)

distorta_filtrato, distorta_Y_filtrato, distorta_spettro_filtrato = filtra_picco_principale(
    distorta_Y_shift,
    freq_distorta,
    distorta_samplerate,
    f0_distorta,
    "Distorta",
    "distorta_filtrata.wav",
    banda=30
)

============================================================
# DOMANDA:
# Perché uso un filtro passa-banda centrato sul picco principale?
#
# RISPOSTA:
# L'obiettivo è mascherare/eliminare tutti i picchi tranne quello principale.
# Per farlo scelgo un filtro passa-banda centrato sulla frequenza f0,
# dove f0 è la frequenza del picco principale trovata prima.
#
# Esempio:
# se f0 = 440 Hz e semi_banda = 30 Hz,
# tengo solo le frequenze tra:
#
# 440 - 30 = 410 Hz
# 440 + 30 = 470 Hz
#
# Quindi la maschera tiene la banda:
#
# 410 Hz <= |f| <= 470 Hz
#
# Uso np.abs(frequenze) perché il segnale audio nel tempo è reale,
# quindi lo spettro della FFT è simmetrico:
# se esiste una componente a +440 Hz, esiste anche la copia a -440 Hz.
# Per ricostruire correttamente il segnale filtrato con la IFFT,
# devo mantenere sia la componente positiva sia quella negativa.
#
# ============================================================
# DOMANDA:
# Questo elimina davvero solo il picco principale?
#
# RISPOSTA:
# Non perfettamente. Il filtro passa-banda con una semi_banda fissa
# non tiene "solo un punto", ma tutta una zona di frequenze attorno a f0.
#
# Quindi, se nella banda f0 ± 30 Hz cade anche un picco secondario vicino,
# anche quel picco secondario verrà mantenuto.
#
# Esempio:
# picco principale: 440 Hz
# picco secondario vicino: 455 Hz
#
# con semi_banda = 30 Hz tengo 410-470 Hz,
# quindi anche 455 Hz viene mantenuto.
#
# Per questo motivo è più corretto dire che:
# "il filtro mantiene la banda attorno al picco principale"
# e non che:
# "il filtro mantiene esclusivamente il picco principale".
#
# ============================================================
# DOMANDA:
# Perché allora uso comunque questo metodo? se cè un picco secondari ocon frequenze simili, io vado a prendere ogni frequenza tra f0-30 e f0 +30 ma potrebbero cadermi True della maschera appartenenti a tutt' alto picco
#
# RISPOSTA:
# Perché la consegna chiede un filtro semplice, scegliendo tra
# passa-basso, passa-alto o passa-banda.
# In questo caso il passa-banda è la scelta più naturale, perché voglio
# mantenere la componente dominante attorno al picco principale
# e azzerare le frequenze più lontane.
#
# Il valore 30 Hz è una scelta pratica/arbitraria:
# è una semi-banda, cioè quanti Hz tengo a sinistra e a destra di f0.
# La banda totale mantenuta è quindi larga 60 Hz.
#
# Se voglio un filtro più selettivo, posso ridurre semi_banda,
# ad esempio usando 10 Hz o 5 Hz.
#
# In alternativa, invece di usare una semi_banda fissa,
# potrei usare la larghezza del picco principale già calcolata con
# peak_widths, ad esempio:
#
# semi_banda = larghezza_f0 / 2
#
# In questo modo la banda del filtro dipende dalla larghezza reale
# del picco principale, ed è meno arbitraria.
# ============================================================


# Il segnale audio nel dominio del tempo è reale:
# i campioni del file audio sono valori reali di ampiezza, ad esempio 0.1, -0.3, 0.5, ...
#
# Quando calcolo la FFT, passo dal dominio del tempo al dominio delle frequenze.
# Le frequenze associate ai coefficienti FFT sono valori reali in Hz:
# ad esempio -440 Hz, 0 Hz, +440 Hz, ecc.
#
# I coefficienti della FFT, invece, possono essere numeri complessi.
# Un coefficiente può essere del tipo a + bj, dove:
# - la parte reale e immaginaria descrivono ampiezza e fase della componente frequenziale
# - il modulo |a + bj| = sqrt(a^2 + b^2) indica quanto pesa quella frequenza
# - la potenza è |a + bj|^2 = a^2 + b^2
#
# Poiché il segnale audio originale è reale, lo spettro della FFT è simmetrico:
# la componente a frequenza negativa è il coniugato della componente positiva.
#
# In formula:
# X[-f] = coniugato(X[f])
#
# Quindi, se a +440 Hz ho un coefficiente 4 + 6j,
# a -440 Hz avrò il coefficiente 4 - 6j.
#
# Per questo motivo lo spettro di potenza è specchiato:
# |X[-f]|^2 = |X[f]|^2.
#
# Musicalmente, però, le frequenze negative non rappresentano nuove note:
# sono una conseguenza matematica della FFT di un segnale reale.
# Per identificare le note considero quindi solo le frequenze positive.

# Inoltre anche parte reale e parte immaginaria dei coefficienti FFT
# mostrano una simmetria particolare.
#
# Se il segnale nel tempo è reale, allora:
#
# X[-f] = coniugato(X[f])
#
# Quindi, se:
#
# X[+440 Hz] = 4 + 6j
#
# allora:
#
# X[-440 Hz] = 4 - 6j
#
# Da questo segue che:
#
# Re(X[-f]) = Re(X[f])
# Im(X[-f]) = -Im(X[f])
#
# Quindi:
# - la parte reale è simmetrica rispetto a 0 Hz
# - la parte immaginaria è antisimmetrica, cioè cambia segno
#
# Esempio:
# a +440 Hz: parte reale = 4, parte immaginaria = 6
# a -440 Hz: parte reale = 4, parte immaginaria = -6
#
# Lo spettro di potenza invece è simmetrico uguale, perché:
#
# |4 + 6j|^2 = |4 - 6j|^2 = 4^2 + 6^2