import pandas as pd
import functii as f
import acp.ACP as acp
import grafice as g
import seaborn as sb


tabel = pd.read_csv('./dataIN/dsaddate (1).csv', index_col=0)
print(tabel)

# selectare variabile utile
var = tabel.columns.values[1:]
print(var, type(var))

# selectare etichete observatii
obs = tabel.index.values
print(obs, type(obs))

# extragere matrice X
X = tabel[var].values
print(X, X.shape)

# standardizare matrice observatii si variabile cauzele
X_std = f.standardizare(X)
print(X_std.shape)
# salvare matrice standardizata in fisier CSV
X_std_df = pd.DataFrame(data=X_std,
                        index=obs, columns=var)
print(X_std_df)
X_std_df.to_csv('./dataOUT/Xstd.csv')

# instatiere model ACP
modelACP = acp.ACP(X_std)
alpha = modelACP.getAlpha()
print(alpha)

# realizare grafic varianta explicata de componentele principale
g.componentePrincipale(valoriProprii=alpha)
# g.afisare()

# extragre componente principale
C = modelACP.getComponente()
# salvare componente principale in fisier CSV
compPrin = ['C'+str(j+1) for j in range(C.shape[1])]
C_df = pd.DataFrame(data=C, index=obs,
                    columns=compPrin)
print(C_df)
C_df.to_csv('./dataOUT/ComponentePrincipale.csv')

# extragere factori de corelatie
factorLoadings = modelACP.getFactorLoadings()
# conversie catre pandas.DataFrame
factorLoadings_df = pd.DataFrame(data=factorLoadings,
                            index=var, columns=compPrin)
print(factorLoadings_df)
# salvare factor loadings in fisier CSV
# ...
# corelograma factori de corelatie
g.corelograma(matrice=factorLoadings_df,
              titlu='Corelograma factorilor de corelatie')
# g.afisare()

# extragere scoruri
scoruri = modelACP.getScoruri()
scoruri_df = pd.DataFrame(data=scoruri,
            index=obs, columns=compPrin)
# salvare scoruri in fisier CSV
# corelograma scorurilor
g.corelograma(matrice=scoruri_df, titlu='Corelograma scorurilor')
# g.afisare()

# calcul calitatea reprezentarii observatiilor
calcObs = modelACP.getCalObs()
calcObs_df = pd.DataFrame(data=calcObs,
                    index=obs, columns=compPrin)
# salvare in fisier CSV calitatea reprezentarii observatiilor
# TODO
g.corelograma(matrice=calcObs_df, titlu='Corelograma calitatatii reprezentarii observatiilor')
# g.afisare()

# calcul cotributie observatii la varianta axelor (a componentelor principale)
contribObs = modelACP.getContribObs()
contribObs_df = pd.DataFrame(data=contribObs,
                             index=obs, columns=compPrin)
# salvare in fisier CSV cotributie observatii la varianta axelor
# # TODO
g.corelograma(matrice=contribObs_df,
              titlu='Corelograma cotributiei observatii la varianta axelor')
# g.afisare()

# calcul comunalitati
comun = modelACP.getComunalitati()
comun_df = pd.DataFrame(data=comun, index=var, columns=compPrin)
# salvare in fisier CSV comunalitatile
# # TODO
g.corelograma(matrice=comun_df, titlu='Corelograma comunalitatilor')
g.afisare()