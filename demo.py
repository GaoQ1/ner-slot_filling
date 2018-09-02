import numpy as np

import code

a = np.array([[[100, 3, 4], [5, 6, 7]], [[8, 3, 4], [5, 9, 7]]])
# b = np.argmax(a, 1)
# c = np.expand_dims(b, 1)


b = np.sort(a, axis=1)

code.interact(local=locals())