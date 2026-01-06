import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.collections import LineCollection
from dataclasses import dataclass
from scipy.spatial import Delaunay
from typing import Optional, TYPE_CHECKING


if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem


@dataclass
class PlotSettings:
    xlabel: str
    ylabel: str
    lines: LineCollection | None
    figsize: tuple
    s: int
    parent_cmap: str
    shrink: float = 1
    saveFile: str = ''
    title: str | None = None


class EcosystemPlot():
    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem

    @property
    def size(self):
        return self.ecosystem.size
    
    @property
    def member_model_ids(self):
        return self.ecosystem.member_model_ids
    
    # this class only reads the following attributes, it doesn't modify them

    @property
    def points(self):
        return self.ecosystem.grid.points
    
    @property
    def feasible_points(self):
        return self.ecosystem.grid.feasible_points
    
    @property
    def pfractions(self):
        return self.ecosystem.grid.pfractions
    

    # =================================================================== PLOTTING ===================================================================


    # distinta, LLAMADA EXPLICITAMENTE
    def plot_2D_slice(self, model_ids = None, fixed_values = None, parent_cmap: str = 'tab20', s: float | int = 8, 
                      xlabel = None, ylabel = None, figsize: tuple[int, int] = (11,12), 
                      to_plot: Optional[str] = None, show_edge: bool = False, frac_prefix = None, 
                      saveFile: str = '', fractions = None, title: Optional[str] = None) -> None:
        
        """ Plots a 2D slice of the grid. The remaining dimensions of the grid are kept at fixed values.
            Individual organism models of fixed dimensions are those determined by 'prefixes'. Their 
            objective function values are fixed to 'fixed_values'.   
            Points previously found unfeasible are not shown.
        
            prefixes: List of individual organism prefixes to be set at fixed objective function values 
            fixed_values: Fixed values for objective functions of 'prefixes' individual models
            parent_cmap: color map used to show clusters
            to_plot: How to color grid points:
                 'cluster':  Points are colored according to their clusters. Clusters must be previously calculated.  
                             if point feasibility has been run, only feasible points are shown.  
                 'feasible': Points are colored as feasible or unfeasible. Point feasibility must 
                             be previously calculated.
                             
                  'community_growth': Points are colored according to their corresponding community growth rate.
                  'member_fraction' : The community fraction of a particular member (frac_prefix) is used to color grid points
                  
                  None: no particular coloring over grid points. 
            
            xlabel: x label. If None member prefix is used
            ylabel: y label. If None member prefix is used
            
            s: marker size
            figsize: figure size
            show_edge: Draw grid edge.
            
            Returns:       
            slice_points: grid points in slice.
        """    

        model_ids = [] if model_ids is None else model_ids
        fixed_values = [] if fixed_values is None else fixed_values

        full_slice_indexes, free_member_indexes, free_member_model_ids = self.ecosystem.grid.resolve_2D_slice(model_ids, fixed_values)
        full_slice_points = self.points[full_slice_indexes,:][:,free_member_indexes]
        
        # get slice feasible points and their member distribution:
        slice_points, slice_pfractions, slice_indexes = self.ecosystem.grid.get_slice_points(full_slice_indexes, free_member_indexes)

        # plot settings    
        xlabel = free_member_model_ids[0] if xlabel is None else xlabel
        ylabel = free_member_model_ids[1] if ylabel is None else ylabel
        lines = self._draw_slice_edge(full_slice_points) if show_edge else None # get edge of full slice
        plot_settings = PlotSettings(xlabel=xlabel, ylabel=ylabel, lines=lines, figsize=figsize, s=s, parent_cmap=parent_cmap, shrink=0.5, saveFile=saveFile, title=title)

        # different plot coloring cases

        # 1. nothing is colored
        if to_plot is None:
            self._no_coloring_plot(slice_points, plot_settings)
             
        # 2. points are colored as feasible and unfeasible        
        elif to_plot == 'feasible':
            self._feasible_plot(full_slice_indexes, full_slice_points, plot_settings)

        # 3. points are colored according to their community growth values    
        elif to_plot == 'community_growth':
            self._community_growth_plot(slice_indexes, slice_points, plot_settings)
            
        # 4. points are colored according to their clusters 
        elif to_plot == 'cluster':
            self._cluster_plot(slice_indexes, slice_points, plot_settings, fractions = fractions)

        # 5. points are colored according to a selected member community fraction
        elif to_plot == 'member_fraction':
            self._member_fraction_plot(slice_pfractions, frac_prefix, slice_points, plot_settings)

        else:
            raise ValueError(f'"{to_plot}" not recognized')
            
           
    @staticmethod
    def _draw_slice_edge(full_slice_points) -> LineCollection:
        #Delaunay triangulation para slice
        slice_delaunay = Delaunay(full_slice_points)        
        
        #Puntos en el borde de slice:
        edges = set()
        edge_points = []   
        for i, j in slice_delaunay.convex_hull:
            if (i, j) not in edges and (j, i) not in edges:
                edges.add( (i, j) )
                edge_points.append(slice_delaunay.points[ [i, j] ])
        
        #plot de borde de slice
        lines = LineCollection(edge_points, color='dimgrey')    
        return lines  
                 

    @staticmethod
    def _no_coloring_plot(slice_points, settings: PlotSettings):
        
        myfig = plt.figure(figsize = settings.figsize)            
        if settings.lines is not None:
            plt.gca().add_collection(settings.lines)    
            
        ps = plt.scatter(slice_points[:,0],slice_points[:,1],s=settings.s) 
        plt.xlabel(settings.xlabel)
        plt.ylabel(settings.ylabel)
        if len(settings.saveFile)>0:
            plt.savefig(str(settings.saveFile)+'.pdf')
        plt.show()


    def _feasible_plot(self, full_slice_indexes, full_slice_points, plot_settings: PlotSettings):
        if self.feasible_points is None:
            raise RuntimeError('Points feasibility analysis has not been run! Required for plot!')
                 
        aux = np.array(self.feasible_points)[full_slice_indexes]
        slice_colors = aux + 1
        k = 2
        color_labels = ['','unfeasible', 'feasible']
            
        self._categorical_coloring_plot(full_slice_points, slice_colors, k, color_labels, plot_settings)      


    def _community_growth_plot(self, slice_indexes, slice_points, plot_settings: PlotSettings):
        cgrowth = self.ecosystem.grid._calculate_community_growth(feasible=True)
        slice_colors = cgrowth[slice_indexes]
            
        self._gradient_coloring_plot(slice_points, slice_colors, color_label = 'community growth', plot_settings = plot_settings)   

        
    def _cluster_plot(self, slice_indexes, slice_points, plot_settings: PlotSettings, fractions = None):        
        slice_colors = self.ecosystem.clustering.clusters[slice_indexes] #slice points clusters
        k = self.ecosystem.clustering.k            
        color_labels = ['']+['c'+ str(x+1) for x in range(k)] 
        pfrac = self.pfractions
        if fractions is not None:
            fracBool = np.isclose(pfrac, fractions)
            #print(pfrac[fracBool])
            #print(sum(fracBool))
            #fracBool = pfrac == fractions
            frac = [f[0] and f[1] for f in fracBool]
            pointsF = [[self.points[i][0],self.points[i][1]] for i,f in enumerate(frac) if f]
            print(pointsF)
        else:
            pointsF = None

        self._categorical_coloring_plot(slice_points, slice_colors, k, color_labels, plot_settings, pointsF = pointsF)


    def _member_fraction_plot(self, slice_pfractions, frac_prefix, slice_points, plot_settings: PlotSettings):    
            if frac_prefix is None or frac_prefix not in self.member_model_ids:
                 raise RuntimeError('Missing valid member prefix: frac_prefix')
             
            member_index = self.member_model_ids.index(frac_prefix)
            slice_colors = slice_pfractions[:,member_index]
            color_label = "%s fraction" % frac_prefix
            
            self._gradient_coloring_plot(slice_points, slice_colors, color_label, plot_settings)                        

                 
    @staticmethod
    def _categorical_coloring_plot(slice_points, slice_colors, k, color_labels, settings: PlotSettings, pointsF = None):
        myfig = plt.figure(figsize=settings.figsize)     

        if settings.lines is not None: 
            plt.gca().add_collection(settings.lines)

        cmap1=plt.get_cmap(settings.parent_cmap, k)

        vmin = 0.5
        vmax = k + 0.5 

        if pointsF is not None:
            x = [0]+[p[0] for p in pointsF]
            y = [0]+[p[1] for p in pointsF]

            ps = plt.scatter(slice_points[:,0], slice_points[:,1], c = slice_colors, cmap=cmap1, s=settings.s, vmin=vmin, vmax=vmax, alpha=0.5)
            plt.plot(x,y,'k')

        else:
            ps = plt.scatter(slice_points[:,0], slice_points[:,1], c = slice_colors, cmap=cmap1, s=settings.s, vmin=vmin, vmax=vmax)
       
        cbar = myfig.colorbar(ps, ticks=np.arange(k+1), shrink = settings.shrink)        
        cbar.ax.set_yticklabels(color_labels)  
        plt.xlabel(settings.xlabel)
        plt.ylabel(settings.ylabel)
        if len(settings.saveFile)>0:
            plt.savefig(str(settings.saveFile)+'.png')
        if settings.title is not None:
            plt.title(settings.title)

        plt.show()           

    
    @staticmethod
    def _gradient_coloring_plot(slice_points, slice_colors, color_label: str, settings: PlotSettings):
        myfig = plt.figure(figsize=settings.figsize)            
        if settings.lines is not None:
            plt.gca().add_collection(settings.lines)
    
        ps = plt.scatter(slice_points[:,0],slice_points[:,1], c = slice_colors,cmap=settings.parent_cmap,s=settings.s)            
        cbar = myfig.colorbar(ps, shrink=settings.shrink)
        #cbar.ax.set_ylabel(color_label, rotation=270)
        cbar.set_label(color_label, rotation=270, labelpad= 20)#, labelpad=0.5)
        plt.xlabel(settings.xlabel)
        plt.ylabel(settings.ylabel)
        plt.show()         
    
    
    def plot_qFCA(self, col_wrap = 4):
        #Plots results computed by quan_FCA
        #input: maxmin_df (output of quan_FCA)
        #output: plot
        maxmin_df = self.ecosystem.analyze.qFCA

        sns.set(font_scale = 2)
        rxns_analysis = maxmin_df.columns[0:2]
        sns.set_style("whitegrid")

        g=sns.relplot(data = maxmin_df, x=rxns_analysis[0], y=rxns_analysis[1], col = 'point', hue='FVA', kind='line', col_wrap=4, lw=0)
        points = maxmin_df.point.unique()
        for i,ax in enumerate(g._axes):
            p = points[i]

            p_df = maxmin_df.loc[maxmin_df['point']==p]
            x = p_df.loc[p_df['FVA']=='maximum'][rxns_analysis[0]].to_numpy()

            y1 = p_df.loc[p_df['FVA']=='maximum']
            y1 = y1[rxns_analysis[1]].to_numpy()

            y2 = p_df.loc[p_df['FVA']=='minimum']
            y2 = y2[rxns_analysis[1]].to_numpy()

            ax.fill_between(x, y1,y2, color='none',hatch='//', edgecolor="k", linewidth=0.001)

