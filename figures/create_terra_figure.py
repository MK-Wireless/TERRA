import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, FancyArrowPatch, Arc
from matplotlib.lines import Line2D
import numpy as np
from pathlib import Path

OUT = Path('/mnt/data')

# Canvas at intended two-column print size.
fig, ax = plt.subplots(figsize=(7.15, 2.72))
ax.set_xlim(0, 100)
ax.set_ylim(0, 38)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Palette
NAVY = '#194B7A'
BLUE = '#2878B5'
LBLUE = '#E8F2FA'
GREEN = '#4F8A4C'
LGREEN = '#EAF4E6'
ORANGE = '#D96C2B'
LORANGE = '#FCEBDD'
YELLOW = '#F3C95D'
LYELLOW = '#FFF7D6'
GRAY = '#5B6470'
LGRAY = '#F2F4F6'
DARK = '#1F2933'
RED = '#C64538'

# Helpers

def rbox(x, y, w, h, fc='white', ec=GRAY, lw=0.8, radius=0.9, z=2):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.18,rounding_size={radius}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(p)
    return p


def txt(x, y, s, size=6, weight='normal', color=DARK, ha='center', va='center', z=5, style='normal'):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                   ha=ha, va=va, zorder=z, family='DejaVu Sans', style=style)


def arrow(x1, y1, x2, y2, color=BLUE, lw=1.1, ms=7, rad=0.0, z=3, alpha=1.0):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=ms,
                        linewidth=lw, color=color,
                        connectionstyle=f"arc3,rad={rad}", zorder=z, alpha=alpha)
    ax.add_patch(a)
    return a


def tree(x, y, s=1.0, color=GREEN, z=1):
    ax.add_patch(Rectangle((x-0.22*s, y), 0.44*s, 1.15*s,
                           facecolor='#8B5A2B', edgecolor='none', zorder=z))
    ax.add_patch(Polygon([[x, y+4.2*s],[x-1.4*s,y+1.0*s],[x+1.4*s,y+1.0*s]],
                         closed=True, facecolor=color, edgecolor='white', linewidth=0.25, zorder=z))
    ax.add_patch(Polygon([[x, y+3.2*s],[x-1.1*s,y+1.1*s],[x+1.1*s,y+1.1*s]],
                         closed=True, facecolor=color, edgecolor='white', linewidth=0.25, zorder=z+0.1))


def flame(x, y, s=1.0, z=4):
    outer = Polygon([[x,y],[x-0.9*s,y+1.2*s],[x-0.45*s,y+2.8*s],[x,y+2.2*s],
                     [x+0.35*s,y+3.4*s],[x+0.95*s,y+1.4*s]], closed=True,
                    facecolor=RED, edgecolor='white', linewidth=0.4, zorder=z)
    inner = Polygon([[x,y+0.4*s],[x-0.35*s,y+1.2*s],[x,y+2.0*s],[x+0.35*s,y+1.1*s]],
                    closed=True, facecolor=YELLOW, edgecolor='none', zorder=z+1)
    ax.add_patch(outer); ax.add_patch(inner)


def sensor_node(x, y, label, region, scale=1.0, z=5):
    # device body
    rbox(x-1.7*scale, y-1.4*scale, 3.4*scale, 2.8*scale, fc='#EEF1F4', ec=GRAY, lw=0.7, radius=0.45, z=z)
    ax.add_patch(Rectangle((x-1.2*scale, y-0.8*scale), 2.4*scale, 0.7*scale,
                           facecolor='#B8C4CE', edgecolor='none', zorder=z+1))
    ax.add_patch(Rectangle((x-0.9*scale, y+0.05*scale), 1.8*scale, 0.55*scale,
                           facecolor='#D9E3EA', edgecolor=GRAY, linewidth=0.35, zorder=z+1))
    # antenna
    ax.add_line(Line2D([x+1.0*scale, x+1.0*scale], [y+1.0*scale, y+2.25*scale],
                       color=GRAY, linewidth=0.7, zorder=z+1))
    ax.add_patch(Circle((x+1.0*scale, y+2.35*scale), 0.11*scale, facecolor=BLUE, edgecolor='none', zorder=z+2))
    ax.add_patch(Arc((x+1.0*scale, y+2.35*scale), 1.0*scale, 1.0*scale, theta1=315, theta2=45,
                     edgecolor=BLUE, linewidth=0.6, zorder=z+1))
    txt(x, y-1.95*scale, label, size=4.6, weight='bold', z=z+2)
    ax.add_patch(Circle((x-1.25*scale, y+1.75*scale), 0.24*scale, facecolor=LGREEN, edgecolor=GREEN, linewidth=0.5, zorder=z+2))


def message_bar(x, y, w=9.0, h=1.9, z=7, fontsize=4.1):
    parts = [(0.0, 0.27, LGREEN, GREEN, 'region'),
             (0.27, 0.53, LBLUE, BLUE, 'features'),
             (0.80, 0.20, LYELLOW, '#A77900', 'nonce')]
    cur = x
    for start, frac, fc, ec, lab in parts:
        ww = w*frac
        ax.add_patch(Rectangle((cur, y), ww, h, facecolor=fc, edgecolor=ec, linewidth=0.45, zorder=z))
        txt(cur+ww/2, y+h/2, lab, size=fontsize, color=ec, weight='bold', z=z+1)
        cur += ww


def antenna_tower(x, y, s=1.0, z=6):
    ax.add_line(Line2D([x, x-1.0*s], [y+5.0*s, y], color=GRAY, linewidth=0.9, zorder=z))
    ax.add_line(Line2D([x, x+1.0*s], [y+5.0*s, y], color=GRAY, linewidth=0.9, zorder=z))
    ax.add_line(Line2D([x-0.75*s, x+0.75*s], [y+1.3*s, y+1.3*s], color=GRAY, linewidth=0.65, zorder=z))
    ax.add_line(Line2D([x-0.48*s, x+0.48*s], [y+2.6*s, y+2.6*s], color=GRAY, linewidth=0.65, zorder=z))
    ax.add_patch(Circle((x, y+5.2*s), 0.16*s, facecolor=BLUE, edgecolor='none', zorder=z+1))
    ax.add_patch(Arc((x, y+5.2*s), 1.8*s, 1.8*s, theta1=300, theta2=60, edgecolor=BLUE, linewidth=0.8, zorder=z))
    ax.add_patch(Arc((x, y+5.2*s), 3.0*s, 3.0*s, theta1=305, theta2=55, edgecolor=BLUE, linewidth=0.6, zorder=z))


def neural_icon(x, y, s=1.0, color=NAVY, z=8):
    # tiny 3-layer network
    xs = [x-1.0*s, x, x+1.0*s]
    ys = [[y-0.7*s,y,y+0.7*s],[y-0.5*s,y+0.5*s],[y]]
    for i in range(len(xs)-1):
        for yy1 in ys[i]:
            for yy2 in ys[i+1]:
                ax.add_line(Line2D([xs[i], xs[i+1]],[yy1, yy2], color=color, linewidth=0.35, alpha=0.65, zorder=z))
    for xx, arr in zip(xs, ys):
        for yy in arr:
            ax.add_patch(Circle((xx,yy),0.13*s,facecolor='white',edgecolor=color,linewidth=0.55,zorder=z+1))

# Panel backgrounds
rbox(1.2, 2.0, 39.3, 33.5, fc='#FBFCFD', ec='#C8D0D8', lw=0.8, radius=1.2, z=0)
rbox(42.2, 2.0, 16.1, 33.5, fc='#F9FBFD', ec='#C8D0D8', lw=0.8, radius=1.2, z=0)
rbox(60.0, 2.0, 38.7, 33.5, fc='#FBFCFD', ec='#C8D0D8', lw=0.8, radius=1.2, z=0)

# Panel headings
rbox(2.3, 32.0, 37.1, 2.35, fc=LGREEN, ec=GREEN, lw=0.6, radius=0.7, z=2)
txt(20.85, 33.18, '(a) Task-oriented edge representation', size=7.2, weight='bold', color=GREEN)
rbox(43.2, 32.0, 14.1, 2.35, fc=LBLUE, ec=BLUE, lw=0.6, radius=0.7, z=2)
txt(50.25, 33.18, '(b) URA uplink', size=6.8, weight='bold', color=BLUE)
rbox(61.0, 32.0, 36.7, 2.35, fc=LORANGE, ec=ORANGE, lw=0.6, radius=0.7, z=2)
txt(79.35, 33.18, '(c) Regional evidence aggregation', size=7.2, weight='bold', color=ORANGE)

# Left panel: spatial scene
# Region background zones
ax.add_patch(FancyBboxPatch((2.7, 14.2), 12.0, 15.6, boxstyle='round,pad=0.15,rounding_size=1.0',
                            facecolor='#EDF7EA', edgecolor='#A9C89F', linewidth=0.55, linestyle='--', zorder=0.6))
ax.add_patch(FancyBboxPatch((15.0, 14.2), 11.6, 15.6, boxstyle='round,pad=0.15,rounding_size=1.0',
                            facecolor='#FFF5E8', edgecolor='#DAB78B', linewidth=0.55, linestyle='--', zorder=0.6))
ax.add_patch(FancyBboxPatch((26.9, 14.2), 11.4, 15.6, boxstyle='round,pad=0.15,rounding_size=1.0',
                            facecolor='#EEF5FB', edgecolor='#A7C0D7', linewidth=0.55, linestyle='--', zorder=0.6))
txt(8.7, 28.85, 'Region A', size=5.3, weight='bold', color=GREEN)
txt(20.8, 28.85, 'Region B', size=5.3, weight='bold', color=ORANGE)
txt(32.6, 28.85, 'Region C', size=5.3, weight='bold', color=BLUE)

# Trees and fire
for x,y,s in [(4.2,15.0,0.65),(6.6,19.7,0.72),(11.6,16.2,0.55),(17.1,15.2,0.55),(23.6,19.0,0.7),(28.4,15.4,0.58),(35.7,18.8,0.72),(31.2,22.6,0.52)]:
    tree(x,y,s,color=GREEN,z=1)
flame(20.3,17.0,0.95,z=3)

# Nodes
sensor_node(8.6, 21.2, 'Node 1', 'A', scale=0.9)
sensor_node(20.9, 23.0, 'Node 2', 'B', scale=0.9)
sensor_node(32.4, 20.3, 'Node 3', 'C', scale=0.9)

# Representative local pipeline compact inset
rbox(3.0, 3.2, 35.0, 8.9, fc='white', ec='#B7C1CA', lw=0.7, radius=0.8, z=4)
txt(4.0, 10.95, 'Representative sensor pipeline', size=5.3, weight='bold', color=NAVY, ha='left')
# sensor modalities
rbox(4.0, 5.0, 8.0, 4.7, fc=LGRAY, ec='#B7C1CA', lw=0.55, radius=0.55, z=5)
txt(8.0, 8.75, 'multimodal window', size=4.7, weight='bold', color=GRAY)
for i,(lab,col) in enumerate([('T',RED),('RH',BLUE),('CO',ORANGE),('CO$_2$',NAVY)]):
    ax.add_patch(Circle((5.25+1.75*i,6.75),0.46,facecolor='white',edgecolor=col,linewidth=0.65,zorder=7))
    txt(5.25+1.75*i,6.75,lab,size=4.1,weight='bold',color=col,z=8)
arrow(12.2,7.35,14.0,7.35,color=GRAY,lw=0.8,ms=5)
rbox(14.1,5.0,8.5,4.7,fc=LBLUE,ec=BLUE,lw=0.6,radius=0.55,z=5)
neural_icon(18.35,7.2,0.75,color=NAVY,z=7)
txt(18.35,9.0,'edge encoder',size=4.7,weight='bold',color=NAVY)
txt(18.35,5.55,r'$\mathbf{z}_k=f_\theta(\mathbf{X}_k)$',size=4.4,color=NAVY)
arrow(22.8,7.35,24.4,7.35,color=GRAY,lw=0.8,ms=5)
message_bar(24.55,6.3,w=11.7,h=2.0,z=6,fontsize=3.9)
txt(30.4,9.0,'compact message',size=4.7,weight='bold',color=GRAY)

# Middle panel: uplink
antenna_tower(50.2, 15.2, s=1.65, z=5)
txt(50.2, 27.8, 'common codebook', size=5.5, weight='bold', color=NAVY)
txt(50.2, 25.9, 'concurrent short reports', size=4.8, color=GRAY)
# Arrows from left boundary to tower
for yy,rad in [(23.8,0.09),(20.6,0.02),(17.2,-0.07)]:
    arrow(39.5,yy,48.0,20.6,color=BLUE,lw=1.2,ms=6,rad=rad,z=4)
# URA output arrow to right
arrow(52.6,20.6,59.4,20.6,color=ORANGE,lw=1.3,ms=7,z=5)
rbox(44.1,5.0,12.2,5.0,fc=LBLUE,ec=BLUE,lw=0.6,radius=0.65,z=4)
txt(50.2,8.45,'URA decoder',size=5.5,weight='bold',color=NAVY)
txt(50.2,6.65,'identity-free message list',size=4.7,color=GRAY)

# Right panel: vertical pipeline
# Stage 1
rbox(62.0,26.0,34.5,4.55,fc='white',ec='#BCC4CC',lw=0.65,radius=0.65,z=3)
rbox(62.7,27.0,4.2,2.55,fc=LBLUE,ec=BLUE,lw=0.55,radius=0.45,z=4)
txt(64.8,28.28,'1',size=7.0,weight='bold',color=BLUE)
txt(69.0,29.25,'Recover unordered message multiset',size=5.3,weight='bold',color=NAVY,ha='left')
message_bar(69.2,27.2,w=7.3,h=1.45,z=5,fontsize=3.1)
message_bar(77.3,27.2,w=7.3,h=1.45,z=5,fontsize=3.1)
message_bar(85.4,27.2,w=7.3,h=1.45,z=5,fontsize=3.1)
txt(94.3,27.9,r'$\cdots$',size=7.5,color=GRAY)

# Stage 2
rbox(62.0,20.3,34.5,4.55,fc='white',ec='#BCC4CC',lw=0.65,radius=0.65,z=3)
rbox(62.7,21.3,4.2,2.55,fc=LGREEN,ec=GREEN,lw=0.55,radius=0.45,z=4)
txt(64.8,22.58,'2',size=7.0,weight='bold',color=GREEN)
txt(69.0,23.55,'Group reports by geographic tag',size=5.3,weight='bold',color=NAVY,ha='left')
for i,(lab,fc,ec) in enumerate([('A',LGREEN,GREEN),('B',LORANGE,ORANGE),('C',LBLUE,BLUE)]):
    x=70.0+i*8.3
    rbox(x,21.0,6.5,1.85,fc=fc,ec=ec,lw=0.5,radius=0.35,z=5)
    txt(x+1.0,21.93,f'{lab}',size=4.2,weight='bold',color=ec)
    for j in range(3):
        ax.add_patch(Rectangle((x+2.0+j*1.15,21.35),0.85,1.12,facecolor='white',edgecolor=ec,linewidth=0.35,zorder=6))

# Stage 3
rbox(62.0,13.9,34.5,5.2,fc='white',ec='#BCC4CC',lw=0.65,radius=0.65,z=3)
rbox(62.7,15.2,4.2,2.55,fc=LORANGE,ec=ORANGE,lw=0.55,radius=0.45,z=4)
txt(64.8,16.48,'3',size=7.0,weight='bold',color=ORANGE)
txt(69.0,17.75,'Permutation-invariant regional aggregation',size=5.3,weight='bold',color=NAVY,ha='left')
# DeepSets schematic
for j in range(3):
    ax.add_patch(Circle((71.2+j*1.6,15.8),0.38,facecolor=LBLUE,edgecolor=BLUE,linewidth=0.5,zorder=6))
txt(76.2,15.8,r'$\phi$',size=6.0,weight='bold',color=NAVY)
arrow(74.7,15.8,75.6,15.8,color=GRAY,lw=0.65,ms=4)
arrow(76.8,15.8,79.0,15.8,color=GRAY,lw=0.65,ms=4)
rbox(79.1,14.95,4.3,1.7,fc=LGRAY,ec=GRAY,lw=0.5,radius=0.3,z=5)
txt(81.25,15.8,r'$\sum$',size=7.0,weight='bold',color=GRAY)
arrow(83.6,15.8,85.4,15.8,color=GRAY,lw=0.65,ms=4)
rbox(85.5,14.95,4.3,1.7,fc=LYELLOW,ec='#A77900',lw=0.5,radius=0.3,z=5)
txt(87.65,15.8,r'$\rho$',size=6.5,weight='bold',color='#A77900')
arrow(90.0,15.8,92.4,15.8,color=GRAY,lw=0.65,ms=4)
txt(93.5,15.8,r'$\hat{p}_g$',size=6.0,weight='bold',color=NAVY)

# Stage 4 / Output
rbox(62.0,4.0,34.5,8.6,fc='white',ec='#BCC4CC',lw=0.65,radius=0.65,z=3)
rbox(62.7,9.0,4.2,2.55,fc=LYELLOW,ec='#A77900',lw=0.55,radius=0.45,z=4)
txt(64.8,10.28,'4',size=7.0,weight='bold',color='#A77900')
txt(69.0,11.0,'Region-wise event map',size=5.3,weight='bold',color=NAVY,ha='left')
# outputs
outputs=[('Region A','No fire','0.08',LGREEN,GREEN),('Region B','Fire','0.91',LORANGE,RED),('Region C','No fire','0.14',LBLUE,BLUE)]
for i,(r,status,prob,fc,ec) in enumerate(outputs):
    x=68.8+i*8.8
    rbox(x,5.1,7.6,4.7,fc=fc,ec=ec,lw=0.55,radius=0.55,z=5)
    txt(x+3.8,8.55,r,size=4.4,weight='bold',color=ec)
    if status=='Fire':
        flame(x+1.25,5.85,0.42,z=7)
    else:
        ax.add_patch(Circle((x+1.25,6.75),0.45,facecolor='white',edgecolor=ec,linewidth=0.7,zorder=7))
        ax.add_line(Line2D([x+0.98,x+1.18],[6.75,6.5],color=ec,linewidth=0.75,zorder=8))
        ax.add_line(Line2D([x+1.18,x+1.55],[6.5,7.0],color=ec,linewidth=0.75,zorder=8))
    txt(x+4.4,6.95,status,size=4.7,weight='bold',color=ec)
    txt(x+4.4,5.75,f'$p={prob}$',size=4.2,color=GRAY)

# Direction arrows between stages
for yy1,yy2 in [(25.9,24.9),(20.2,19.2),(13.8,12.75)]:
    arrow(79.3,yy1,79.3,yy2,color=GRAY,lw=0.65,ms=4,z=5)

# Compact footer note
# no title to save space; overall framework name in caption

plt.subplots_adjust(left=0.004, right=0.996, top=0.992, bottom=0.015)

pdf = OUT/'terra_architecture_compact.pdf'
svg = OUT/'terra_architecture_compact.svg'
png = OUT/'terra_architecture_compact.png'
fig.savefig(pdf, bbox_inches='tight', pad_inches=0.015)
fig.savefig(svg, bbox_inches='tight', pad_inches=0.015)
fig.savefig(png, dpi=500, bbox_inches='tight', pad_inches=0.015)
print(pdf)
print(svg)
print(png)
