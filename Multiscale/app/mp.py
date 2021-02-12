import plotly.express as px
import plotly
from skimage import io
img = io.imread('/home/mhbrt/Desktop/Wind/Multiscale/app/mpld3.png')
fig = px.imshow(img)
# fig.show()

plotly.io.write_html(fig, 'sd')
# d=plotly.io.to_html(fig)
# print(d)






# import numpy as np
# import matplotlib.pyplot as plt
# # from mpld3._display import display_d3
# import mpld3
# import numpy as np
# import matplotlib.cm as cm
# import matplotlib.pyplot as plt
# import matplotlib.cbook as cbook
# from matplotlib.path import Path
# from matplotlib.patches import PathPatch

# with cbook.get_sample_data('/home/mhbrt/Desktop/Wind/Multiscale/app/app/static/img/princess.jpg') as image_file:
#     image = plt.imread(image_file)
# image = ""
# fig, ax = plt.subplots()
# ax.imshow(image)
# ax.axis('off')  # clear x-axis and y-axis

# # fig, ax = plt.subplots()
# # np.random.seed(0)
# # ax.plot(np.random.normal(size=100),
# #         np.random.normal(size=100),
# #         'or', ms=10, alpha=0.3)
# # ax.plot(np.random.normal(size=100),
# #         np.random.normal(size=100),
# #         'ob', ms=20, alpha=0.1)

# # ax.set_xlabel('this is x')
# # ax.set_ylabel('this is y')
# # ax.set_title('Matplotlib Plot Rendered in D3!', size=14)
# # ax.grid(color='lightgray', alpha=0.7)

# # plt.show(fig)
# # display_d3(fig)
# # mpld3.show(fig)
# mpld3.fig_to_dict(fig)