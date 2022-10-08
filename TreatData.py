import numpy as np
import pandas as pd

from Implementation import *

data = load_data()
Frame=pd.DataFrame(data)

for nligne, d in enumerate(Frame):
    for ncolumn, j in enumerate(d):
        if (np.allclose(j, -999, atol = 1e-05)):
            np.delete(Frame, [nligne,ncolumn])

print(data)
