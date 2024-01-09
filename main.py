import numpy as np
from sko.GA import GA_TSP
import matplotlib.pyplot as plt
import geopandas as gp
turkiye_iller = gp.read_file('TUR_adm1.shp')
turkiye_iller.plot()


def sehir_ciz(sehirler, rota):
    x = [sehirler[i][0] for i in rota]
    y = [sehirler[i][1] for i in rota]

    plt.plot(x, y, 'o-r', label='Optimal Rota')
    plt.xlabel('X Koordinatı')
    plt.ylabel('Y Koordinatı')
    plt.title('Şehirler ve Optimal Rota')
    plt.legend()
    plt.show()

sehirler = np.array([[30, 41], [30, 39], [32.5, 40], [35, 38], [40, 39], [42.5, 38], [40, 40], [42, 41.5],[35,41],[32.5,38]])

# Mesafeleri hesapla
num_sehirler = len(sehirler)
mesafe_matrisi = np.zeros(shape=(num_sehirler, num_sehirler))
for i in range(num_sehirler):
    for j in range(num_sehirler):
        mesafe_matrisi[i][j] = np.linalg.norm(sehirler[i] - sehirler[j], ord=2)

# Toplam mesafeyi hesapla
def toplam_mesafe_hesap(rota):
    toplam_mesafe = 0
    for i in range(num_sehirler - 1):
        toplam_mesafe += mesafe_matrisi[rota[i], rota[i + 1]]
    toplam_mesafe += mesafe_matrisi[rota[-1], rota[0]]
    print("Toplam mesafe :",toplam_mesafe)
    return toplam_mesafe
ga_tsp = GA_TSP(func=toplam_mesafe_hesap, n_dim=num_sehirler, size_pop=50, max_iter=200, prob_mut=0.01)
optimal_rota, optima_mesafe = ga_tsp.fit()

optimal_rota = np.append(optimal_rota, optimal_rota[0])

sehir_ciz(sehirler, optimal_rota)