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

sehirler = np.array([[30, 41], [30, 39], [32.5, 40], [35, 38], [40, 39], [42.5, 38], [40, 40], [42, 41.5], [35, 41], [32.5, 38]])

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
    return toplam_mesafe

ga_tsp = GA_TSP(func=toplam_mesafe_hesap, n_dim=num_sehirler, size_pop=2, max_iter=100, prob_mut=0.01)

# Her neslin toplam mesafesini saklamak için liste
x = toplam_mesafe_listesi = []

for _ in range(ga_tsp.max_iter):
    optimal_rota, optima_mesafe = ga_tsp.run()
    toplam_mesafe_listesi.append(optima_mesafe)

optimal_rota = np.append(optimal_rota, optimal_rota[0])

sehir_ciz(sehirler, optimal_rota)

# Her neslin toplam mesafesini göster
print("Toplam Mesafe Listesi:", x)

fig, ax = plt.subplots()
ax.step(range(1, len(x) + 1), x, linewidth=2)  # linewidth değerini düzenleyin
ax.set(xlim=(0, len(x) + 10), xticks=np.arange(10, len(x) + 10, 10),
       ylim=(0, max(x) + 10), yticks=np.arange(0, max(x) + 10, 10))

ax.set_xlabel('İterasyon')  # x ekseni etiketi
ax.set_ylabel('Toplam Mesafe')  # y ekseni etiketi

plt.show()