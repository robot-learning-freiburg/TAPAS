def set_ax_border(ax, color, width=None):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)

        if width is not None:
            spine.set_linewidth(2)
