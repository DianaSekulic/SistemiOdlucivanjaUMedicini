# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 22:20:23 2022

@author: Minja
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('02_toddler_autism_dataset.csv')
info = data.info(verbose=True) # da li postoje vrednosti koje nedostaju
data_describe = data.describe().T

# sve ok
#%%

ethnicity_duplicates = data.drop_duplicates(subset = ["Ethnicity"])
test_duplicates = data.drop_duplicates(subset = ["Who completed the test"])
sex_duplicates = data.drop_duplicates(subset = ["Sex"])


#%%

# replace f i m
data['Sex'] = data['Sex'].replace('f', 1)
data['Sex'] = data['Sex'].replace('m', 0)

data['Jaundice'] = data['Jaundice'].replace('yes', 1)
data['Jaundice'] = data['Jaundice'].replace('no', 0)

data['Family_mem_with_ASD'] = data['Family_mem_with_ASD'].replace('yes',1)
data['Family_mem_with_ASD'] = data['Family_mem_with_ASD'].replace('no',0)

data['Who completed the test'] = data['Who completed the test'].replace('family member', 0)
data['Who completed the test'] = data['Who completed the test'].replace('Health Care Professional', 1)
data['Who completed the test'] = data['Who completed the test'].replace('Health care professional', 1)
data['Who completed the test'] = data['Who completed the test'].replace('Self', 2)
data['Who completed the test'] = data['Who completed the test'].replace('Others', 3)

data['Class/ASD Traits '] = data['Class/ASD Traits '].replace('No', 0)
data['Class/ASD Traits '] = data['Class/ASD Traits '].replace('Yes', 1)

data['Ethnicity'] = data['Ethnicity'].replace('middle eastern', 10)
data['Ethnicity'] = data['Ethnicity'].replace('White European', 11)
data['Ethnicity'] = data['Ethnicity'].replace('Hispanic', 12)
data['Ethnicity'] = data['Ethnicity'].replace('black', 13)
data['Ethnicity'] = data['Ethnicity'].replace('asian', 14)
data['Ethnicity'] = data['Ethnicity'].replace('south asian', 15)
data['Ethnicity'] = data['Ethnicity'].replace('Native Indian', 16)
data['Ethnicity'] = data['Ethnicity'].replace('Others', 17)
data['Ethnicity'] = data['Ethnicity'].replace('Latino', 18)
data['Ethnicity'] = data['Ethnicity'].replace('mixed', 19)
data['Ethnicity'] = data['Ethnicity'].replace('Pacifica', 20)
data.to_csv('out.csv')
#%%
# spirmanova korelacija
# pirsonov koristimo kad imamo neki outlajer, a ovde nema outlajer

spearman_R = data.corr(method='spearman')
plt.figure(figsize=(16, 16))
sns.heatmap(spearman_R, annot=True)
plt.show()

# mini problem sa korelacijama jeej

def izracunajR(cor_mat):
    k = cor_mat.shape[0] - 1 #posto je kvadratna matrica, a minus jedan zbog klase
    #print(k)
    rzi = np.mean(cor_mat.iloc[:-1,-1])
    #print(rzi)
    rij = np.mean(np.mean(cor_mat.iloc[:-1,-1])) - 1/k
    #print(rij)
    
    r = k*rzi/(k+(k-1)*rij)
    return r

r_ukupno = izracunajR(spearman_R)
print(r_ukupno)

for kk in range(data.shape[1]-1):
    data_1 = data.drop(data.columns[kk], axis = 1)
    corr_1 = data_1.corr(method = 'spearman')
    r = izracunajR(corr_1)
    print(data.columns[kk] + ' ' + str(r))
    print('---------')
    
# ako hocy poslednju kolonu class data.iloc[:,-1]

#%% IG PAZI NISI IZABRALA 10 VEC SVE

def calculateInfoD(kol): # racunam samo za jedinstvene vrednosti, ne za one koji se
    jed_vr = np.unique(kol) # ponavljaju
    infoD = 0
    for jv in jed_vr:
        p = sum(kol == jv) / len(kol) #jed_vr su jedinstvene i to sluzi samo za for petlju
        infoD -= p*np.log2(p)       # ovde su svi jed_vr u koloni poslatoj i trazim procentualno
                                    # suma koliko njih ima taj izlaz(klasu npr 0) i delim sa uk duzinom kolone poslate
    return infoD

klasa = data.iloc[:,-1] # gledamo posl, to je outcome, oznaka klase, da li pripada ili ne
# za klasu racunamo Info, moze #data.iloc[8] data.iloc[-1] data.Outcome
infoD = calculateInfoD(klasa)
print('Info(D) = ' + str(infoD))
new_data = data.copy(deep=True)


def limit_feature(kol, brojKoraka=30): #po deafultu 30 bins-a u histogramu
    korak = (max(kol)-min(kol))/brojKoraka # da bi izbegli outliere i guess :/
    nova_kol = np.floor(kol/korak)*korak # ovo radimo da imamo ekvidistantne tacke/vrednosti
    return nova_kol

for ob in range(1, data.shape[1]-1): #data.shape daje broj elem (-1 ide za outcome)
    temp = data.iloc[:, ob]         #data.shape[1] je koliko ima kolona (druga dim. za 2D slucaj)
    new_data.iloc[:, ob] = limit_feature(temp) #data.shape[0] je koliko ima vrsta (prva dim)
    # data.iloc[0] je prva vrsta, data.iloc[0,1] je podatak prva vrsta druga kolona
    # sto ob od 1,a ne od 0? TIme izbacujemo pregnancies i izbacujemo outcome
    
    
IG = np.zeros((new_data.shape[1]-1, 2))
for ob in range(new_data.shape[1]-1):
    f = np.unique(new_data.iloc[:, ob])
    infoDA = 0
    for i in f:
        temp = klasa[new_data.iloc[:, ob] == i]
        infoDi = calculateInfoD(temp)
        Di = sum(new_data.iloc[:, ob] == i)
        D = len(new_data.iloc[:, ob])
        infoDA += Di*infoDi/D
    print('Info(D/A) = ' + str(infoDA))
    print('------')
    IG[ob, 0] = ob+1
    IG[ob, 1] = infoD - infoDA
    
# print('IG = ' + str(IG))
IGsorted = IG[IG[:, 1].argsort()]
# print('Sortirano IG = ' + str(IG))

#%% LDA na 1 dim LOOL OVO NEMA SMISLA AHAHAHHA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

X_pom = data.iloc[:,:-1]

X = X_pom - np.mean(X_pom, axis = 0)
X /= np.max(X, axis = 0)

#izlaz = data.iloc[:,-1]

lda = LDA(n_components = 1) # maksimalno mozemo da izaberemo broj_klasa-1 komponenti koje uzimamo (ili sve komponente)
Y2_lda = lda.fit_transform(X, klasa)

data_LDA2 = pd.DataFrame(data = Y2_lda)
data_LDA2 = pd.concat([data_LDA2, klasa], axis = 1)

plt.figure()
sns.scatterplot(data = data_LDA2, x = 0 , y = 1, hue = 'Class/ASD Traits ') # hue boji parametre
plt.show()

# PCA NA 2 DIM
from sklearn.decomposition import PCA

# na projektu mozemo da koristimo gotove funkcije, ali treba da znamo kako funkcionisu 

pca = PCA(n_components = 2) #probamo prvo sa 2 pa promenimo na 3, a onda mora i plot da se promeni
Y2_pca = pca.fit_transform(X)
# kad se radi sa gotovom funkcijom moze da se dobije negde isti broj a drugi znak kao kad radimo rucno ali to je ok jer znak nije bitan

data_PCA2 = pd.DataFrame(data = Y2_pca)
data_PCA2 = pd.concat([data_PCA2, klasa], axis = 1) # axis = 1 zbog spajanja po kolonama

#plt.figure()
#sns.scatterplot(data = data_PCA2, x = 0 , y = 1, hue = 'Class') # hue boji parametre

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter( Y2_pca[:,0],  Y2_pca[:,1], c=klasa)

plt.show()

# komentar - izgleda da bolje PCA daje rezultate ali Hesus Maria sta je ovo
# ne znam da citam grafike hesus lord...i ne znam razlog sto je bolja...

#%%
# TACKA 7. PROJEKTOVANJJE LIN KLASIF KORISTECI PODATKE IZ LDA METODE
# K1 ce biti klasa za 0, a K2 klasa za 1 
K1 = data_LDA2[data_LDA2.iloc[:,-1]==0]
K1 = K1.iloc[:,0]
K1 = K1.to_numpy()
K1 = K1.reshape(K1.size, 1)
K1 = K1.T
K2 = data_LDA2[data_LDA2.iloc[:,-1]==1]
K2 = K2.iloc[:,0]
K2 = K2.to_numpy()
K2 = K2.reshape(K2.size, 1)
K2 = K2.T

Z1 = np.append(-K1, -np.ones((1, len(K1.T))), axis=0)
Z2 = np.append(K2, np.ones((1, len(K2.T))), axis=0)
U = np.append(Z1, Z2, axis=1)

Gama = np.ones((len(U.T), 1))
W = np.linalg.inv(U@U.T)@U@Gama
V = W[:-1]
V0 = W[-1]
V0 = V0.reshape(V0.size,1)

x = -V/V0
X = [x[0], x[0]]
Y = [-5, 30]

plt.figure()
sns.scatterplot(data = data_LDA2, x = 0 , y = 1, hue = 'Class/ASD Traits ') # hue boji parametre
plt.plot(X,Y)
plt.show()