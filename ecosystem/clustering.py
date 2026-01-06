import numpy as np
import pandas as pd

import matplotlib.ticker as mtick

from scipy.spatial import distance
from sklearn.cluster import DBSCAN, OPTICS,SpectralClustering, AffinityPropagation
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem


class EcosystemClustering():
    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem
        self.k, self.rk = None, None
        self.clusters, self.rclusters = None, None
        self.bin_vector_df = None

    @property
    def qual_vector_df(self): # this class only reads this attribute, it doesn't modify it
        return self.ecosystem.analyze.qual_vector_df


    # NO SE USA EN NINGUN MODULO
    def clusterReactions(self, method, changing= True, **kwargs):

        if self.qual_vector_df is None:
            print("No qualitative FVA values stored. Run qual_fva analysis first!")
            return

        print("Calculating jaccard distances between grid points...") 
        
        z = self.qual_vector_df.copy()
        self.changed_rxns = None
        if changing: #clustering considering changing reactions only:
            changed_rxns = self.qual_vector_df.max(axis=0) != self.qual_vector_df.min(axis=0)
            changed_rxns_ids = z.columns[changed_rxns]
            z = z[changed_rxns_ids]
            self.changed_rxns = changed_rxns_ids
            
        z = z.values
        z = z.T
        nrxns = z.shape[0]
        distance_metric = 'jaccard'
        dvector = distance.pdist(z,distance_metric)
        dmatrix = distance.squareform(dvector)
        
        print("Clustering %d reactions ..." % nrxns) 
        # Clustering methods
        # hierarchical + fcluster
        # dbscan (eps,min_samples)
        # optics
        # AffinityPropagation
        # SpectralClustering 
       
        if method == 'hierarchical':
            rk, rclusters = getHIERARCHICALclusters(dvector,**kwargs)
        elif method == 'dbscan':
            rk, rclusters = getDBSCANclusters(dmatrix,**kwargs)  
        elif method == 'optics':
            rk, rclusters = getOPTICSclusters(dmatrix,**kwargs)
        elif method == 'SpectralClustering':
            rk, rclusters = getSCclusters(dmatrix,**kwargs)          
        elif method == 'AffinityPropagation':
            rk, rclusters = getAPclusters(dmatrix, **kwargs)
            
        self.rk = rk
        self.rclusters = rclusters    
        print("Done!")    


    # SE LLAMA EXPLICITAMENTE    
    def clusterPoints(self, method, numeric_delta=1e-4, vector = 'qual_vector', run_fva = True , **kwargs):
  
        if self.qual_vector_df is None:
            print("No qualitative FVA values stored. Run qual_fva analysis first!")
            return

        print("Calculating jaccard distances between grid points...") 
        
        #NJT to use qual_vector as well as bin_vector if required
        if vector =='qual_vector':
            z = self.qual_vector_df.values 
        elif vector == 'bin_vector' and self.bin_vector_df is not None:
            z = self.bin_vector_df.values
        
        distance_metric = 'jaccard'
        dvector = distance.pdist(z,distance_metric) # np.ndarray
        dmatrix = distance.squareform(dvector)      # np.ndarray
        
        print("Clustering grid points ...") 
        # Clustering methods
        # hierarchical + fcluster
        # dbscan (eps,min_samples)
        # optics
        # AffinityPropagation
        # SpectralClustering 
       
        if method == 'hierarchical':
            k, clusters = getHIERARCHICALclusters(dvector,**kwargs)
        elif method == 'dbscan':
            k, clusters = getDBSCANclusters(dmatrix,**kwargs)  
        elif method == 'optics':
            k, clusters = getOPTICSclusters(dmatrix,**kwargs)
        elif method == 'SpectralClustering':
            k, clusters = getSCclusters(dmatrix,**kwargs)          
        elif method == 'AffinityPropagation':
            k, clusters = getAPclusters(dmatrix, **kwargs)
            
        self.k = k
        self.clusters = clusters    
        print("Done!")  


    # SE USA EN AMBOS MODULOS
    def get_cluster_reaction_values(self, vector = 'qual_vector', thr=0.75, changing= True, convert=True):
        
        if self.clusters is None:
            raise RuntimeError("Missing clustering/qualitative FVA results!")
        
        #function to get representative qualitative values of a reaction in a cluster
        def get_rep_vals(x,thr):
            rep_val = None
            total = len(x)
            qual_vals, counts = np.unique(x, return_counts=True)
            rep = qual_vals[counts/total >= thr]
         
            if rep.size > 0:
                return rep[0]   #qualitative value present if at least thr of reactions in cluster  
            return None       
               
        #NJT to use qual_vector as well as bin_vector if required
        if vector =='qual_vector'and self.qual_vector_df is not None:
            z = self.qual_vector_df
        elif vector == 'bin_vector' and self.bin_vector_df is not None:
            z = self.bin_vector_df
        
        vector_df = z.astype('int32')
        
        cluster_ids = np.arange(1,self.k+1)
        cluster_dfs = [vector_df[self.clusters == c] for c in cluster_ids]
        aux = [ df.apply(get_rep_vals, thr=thr) for df in cluster_dfs]
        reps = pd.concat(aux, axis=1)
        reps.columns = ['c%d' % x for x in cluster_ids]

        reps = reps.astype('float')
        
        if changing: #report only reactions that have different qualitative values in at least two clusters
            changing_filter = reps.apply(lambda x: x.unique().size > 1, axis = 1)    
            reps = reps[changing_filter.values]
        
        if convert:
            cat_dict = {-3.0: '-', -2.0: '--',-1.0: '-0',1.0: '0+',0.0: '0',2.0: '++',3.0: '+',4.0: '-+',5.0: 'err',100.0: 'var'}
            reps = reps.replace(cat_dict)

        return reps


    # SE USA EN AMBOS MODULOS
    @staticmethod
    def compare_clusters(clusters_df, cid_1, cid_2):
        
        #juice
        if isinstance(cid_1, int):
            cid_1 = 'c%d' % cid_1
        if isinstance(cid_2, int):
            cid_2 = 'c%d' % cid_2            
        
        df = clusters_df[[cid_1,cid_2]]
        changing_filter = df.apply(lambda x: x.unique().size > 1, axis = 1) 
        df = df[changing_filter.values]
        return df
    

    # NO SE USA EN NINGUN MODULO    
    @staticmethod
    def plot_cluster_distribution(clusters_df, cmap='Accent',figsize=(10,5)):
        cl_columns = list(clusters_df)
        cat_percents_dict = dict()
        nreactions = clusters_df.shape[0]
        nan_rep = 100.0
        #1. Change nan to additional category 'variable'
        df2 = clusters_df.fillna(nan_rep)
        
        
        for c in cl_columns:
            vc = df2[c].value_counts()
            
            vc = 100 * vc/nreactions #to percentages
            cat_percents_dict[c] =  vc.to_dict()   
            
        cat_percents = pd.DataFrame.from_dict(cat_percents_dict, orient='index')
        cat_percents.fillna(0, inplace=True)
        
        cat_dict = {-3.0: '-',
                    -2.0: '--',
                    -1.0: '-0',
                     1.0: '0+',
                     0.0: '0',
                     2.0: '++',
                     3.0: '-+',
                     4.0: 'err',
                     5.0: '+',
                     100.0: 'var'}
        
        cat_percents.rename(columns = cat_dict, inplace=True)
        #plot
        ax = cat_percents.plot.barh(stacked=True, cmap=cmap,figsize=figsize)
        ax.legend(loc='center left',bbox_to_anchor=(1.0, 0.5),title='reaction category');
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel('clusters')
        ax.set_xlabel('reactions')

        return cat_percents 
    


# ======================================================= CLUSTER FUNCTIONS =======================================================


def getHIERARCHICALclusters(dvector: np.ndarray, k: int = 20, lmethod: str = 'ward', 
                            criterion: str= 'maxclust', **kwards) -> tuple[int, np.ndarray]:
    row_linkage = hierarchy.linkage(dvector, method=lmethod)
    clusters = fcluster(row_linkage, k, criterion=criterion)

    return k, clusters # clusters are indexed from 1


# retorna un vector de clusters 0-indexados
def getDBSCANclusters(dmatrix: np.ndarray, eps: float = 0.05, 
                      min_samples: int = 5, **kwards) -> tuple[int, np.ndarray]: 
    
    # eps : 
    # The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
    # This is not a maximum bound on the distances of points within a cluster. This is the most important 
    # DBSCAN parameter to choose appropriately for your data set and distance function.
 
    # min_samples:
    # The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. 
    # This includes the point itself.
    
    dbscan = DBSCAN(eps=eps, min_samples = min_samples, metric='precomputed', **kwards)
    clusters = dbscan.fit_predict(dmatrix) # np.ndarray con las etiquetas de cada punto
    
    # output incluye outliers, valor -1 en vector clusters
    k = len(np.unique(clusters))
    
    return k, clusters # clusters are indexed from 0, includes outliers as -1
    

def getOPTICSclusters(dmatrix: np.ndarray, max_eps: float = 0.05, 
                      min_samples: int = 5, **kwards) -> tuple[int, np.ndarray]:
    
    optics = OPTICS(max_eps = max_eps, min_samples = min_samples,metric='precomputed')
    clusters = optics.fit_predict(dmatrix)

    # output incluye outliers, valor -1 en vector clusters
    k = len(np.unique(clusters))

    return k, clusters # clusters are indexed from 0, includes outliers as -1


def getSCclusters(dmatrix: np.ndarray, assign_labels: str = "discretize",
                  random_state: int = 0, k: int = 20, delta: float = 0.2, 
                  **kwards) -> tuple[int, np.ndarray]:
    
    #transformación de matriz de distancia a matriz de similitud. Vía aplicación de Gaussian (RBF, heat) kernel:
    sim_matrix = np.exp(- dmatrix ** 2 / (2. * delta ** 2))
    
    sc = SpectralClustering(n_clusters = k, assign_labels = assign_labels, 
                            random_state = random_state, affinity = 'precomputed', **kwards)
    clusters = sc.fit_predict(sim_matrix)

    return k, clusters # clusters are indexed from 0


def getAPclusters(dmatrix: np.ndarray, **kwards) -> tuple[int, np.ndarray]:

    af = AffinityPropagation(**kwards)
    clusters = af.fit_predict(dmatrix) # np.ndarray

    # output incluye outliers, valor -1 en vector clusters
    k = len(np.unique(clusters))

    return k, clusters # clusters are indexed from 0, includes outliers as -1


def relabel_clusters(cluster_array: np.ndarray) -> np.ndarray:
    pass
