import pandas as pd
from apyori import apriori

MusicData = pd.read_csv('MusicPreference.csv', header=None)

Warehouse = []
for i in range(0, 19):
    Warehouse.append([str(MusicData.values[i, j]) for j in range(10)])
    associations = apriori(Warehouse, min_length=2, min_support=0.2, min_confidence=0.2, min_lift=2)
    associations_result = list(associations)

print(MusicData)
print(associations_result)
