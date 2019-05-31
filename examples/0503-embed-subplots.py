import pymisca.vis_util as pyvis
plt  = pyvis.plt
import pymisca.ext as pyext
np = pyext.np

fig = plt.gcf()
left = 0.05
bottom = 0.05 
width = 0.9
height = 0.9
ax = fig.add_axes([left, bottom, width, height])
# 
# vars(ax)

ax.plot((ys))
ax = pyvis.axis__subaxis(ax,[0.1,0.1,0.9,0.9])
ax.plot((ys))

ax = pyvis.axis__subaxis(ax,[0.2,0.2,0.8,0.8])
ax.plot((ys))

ax = pyvis.axis__subaxis(ax,[0.2,0.2,0.8,0.8])
ax.plot((ys))

ax = pyvis.axis__subaxis(ax,[0.2,0.2,0.8,0.8])
ax.plot((ys))