from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
import matplotlib.ticker as mtick

from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy
from sklearn.cluster import DBSCAN, OPTICS, SpectralClustering, AffinityPropagation

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem


class EcosystemClustering():
    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem
        self.grid_n_clusters: int = 0
        self.reaction_n_clusters: int = 0
        self.grid_clusters, self.reaction_clusters = None, None
        self.bin_vector_df = None


    @property
    def qual_vector_df(self) -> pd.DataFrame:
        return self.ecosystem.analyze.qual_vector_df


    # NO SE USA EN NINGUN MODULO
    def set_reaction_clusters(self, method, changing= True, **kwargs) -> None:
        """Cluster reactions based on their qualitative FVA profiles across grid points.

        Optionally restricts clustering to reactions whose qualitative state changes
        across the grid. Stores cluster labels and number of clusters as attributes.
        """
        if self.qual_vector_df is None:
            print("No qualitative FVA values stored. Run qual_fva analysis first!")
            return

        z = self.qual_vector_df.copy()
        self.changed_rxns = None
        if changing: # clustering considering changing reactions only:
            changed_rxns = self.qual_vector_df.max(axis=0) != self.qual_vector_df.min(axis=0)
            changed_rxns_ids = z.columns[changed_rxns]
            z = z[changed_rxns_ids]
            self.changed_rxns = changed_rxns_ids
            
        z = z.values
        z = z.T

        print(f"Clustering {z.shape[0]} reactions ...") 
        self.reaction_n_clusters, self.reaction_clusters = self._map_clusters(method, z, **kwargs)

 
    def set_grid_clusters(self, method: str, vector: str = 'qual_vector', **kwargs) -> None:
        """Cluster grid points based on qualitative or binary flux vectors.

        Uses pairwise Jaccard distances between grid points and stores the resulting
        cluster labels and number of clusters as attributes.
        """
        if self.qual_vector_df is None:
            print("No qualitative FVA values stored. Run qual_fva analysis first!")
            return

        #NJT to use qual_vector as well as bin_vector if required
        if vector == 'qual_vector':
            qualitative_vector = self.qual_vector_df.values 
        elif vector == 'bin_vector' and self.bin_vector_df is not None:
            qualitative_vector = self.bin_vector_df.values
        else:
            raise ValueError(f"Unknown vector: {vector}")

        print("Clustering grid points ...") 
        self.grid_n_clusters, self.grid_clusters = self._map_clusters(method, qualitative_vector, **kwargs)


    @staticmethod
    def _map_clusters(method: str, qualitative_vector: np.ndarray, **kwargs) -> tuple[int, np.ndarray]:
        """Compute pairwise Jaccard distances and apply a clustering method.

        Returns the number of clusters and a vector of cluster labels.
        """
        distance_metric = 'jaccard'
        dvector = distance.pdist(qualitative_vector, distance_metric) 
        dmatrix = distance.squareform(dvector)     
        
        # Clustering methods
        # hierarchical + fcluster
        # dbscan (eps,min_samples)
        # optics
        # AffinityPropagation
        # SpectralClustering 
       
        if method == 'hierarchical':
            n_clusters, clusters = get_hierarchical_clusters(dvector,**kwargs)
        elif method == 'dbscan':
            n_clusters, clusters = get_DBSCAN_clusters(dmatrix,**kwargs)  
        elif method == 'optics':
            n_clusters, clusters = get_OPTICS_clusters(dmatrix,**kwargs)
        elif method == 'SpectralClustering':
            n_clusters, clusters = get_SC_clusters(dmatrix,**kwargs)          
        elif method == 'AffinityPropagation':
            n_clusters, clusters = get_AP_clusters(dmatrix, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        

        print(f"Done! n_clusters: {n_clusters}, clusters: {clusters}")    
        
        return n_clusters, clusters


    #function to get representative qualitative values of a reaction in a cluster
    @staticmethod
    def _get_representative_qualitative_values(cluster_column: pd.Series, threshold: float) -> int | None:
        """Return the dominant qualitative value in a cluster if it exceeds a frequency threshold.
        If the threshold is not met, returns None."""
        total = len(cluster_column)

        qualitative_values, counts = np.unique(cluster_column, return_counts=True)
        print(f'thresholds: {counts/total}, qualitative_values: {qualitative_values, counts, total}')
        representative = qualitative_values[counts/total >= threshold]
         
        if representative.size > 0:
            return representative[0] # qualitative value present if at least threshold of reactions in cluster  
        
        return None    


    def get_grid_cluster_qual_profiles(self, vector: str = 'qual_vector', threshold: float = 0.75,
                                     changing: bool = True, convert: bool = True) -> pd.DataFrame:
        """Compute representative qualitative reaction profiles for each grid cluster.

        For each grid cluster, assigns a qualitative value to each reaction if it
        appears in at least a given fraction of grid points. Optionally filters
        reactions that change between clusters and converts qualitative codes.
        """
        if self.grid_clusters is None:
            raise RuntimeError("Missing clustering/qualitative FVA results!")
        
        #NJT to use qual_vector as well as bin_vector if required
        if vector =='qual_vector'and self.qual_vector_df is not None:
            vector_df = self.qual_vector_df
        elif vector == 'bin_vector' and self.bin_vector_df is not None:
            vector_df = self.bin_vector_df
        else:
            raise ValueError(f"Unknown vector: {vector}")
        
        vector_df = vector_df.astype('int32')
        #vector_df.head(200)
        cluster_ids = np.arange(1, self.grid_n_clusters + 1)
        #print(f"cluster_ids: {cluster_ids}, grid_clusters: {self.grid_clusters}")
        cluster_dfs = [vector_df[self.grid_clusters == cluster_id] for cluster_id in cluster_ids]
        print(f"cluster_dfs len: {len(cluster_dfs)}")
        representatives_list = [cluster_df.apply(self._get_representative_qualitative_values, 
                                                 threshold=threshold) for cluster_df in cluster_dfs]
        
        representatives = pd.concat(representatives_list, axis=1).astype('float')
        representatives.columns = [f'c{cluster_id}' for cluster_id in cluster_ids]

        if changing: # report only reactions that have different qualitative values in at least two clusters
            changing_filter = representatives.apply(lambda x: x.unique().size > 1, axis = 1)    
            representatives = representatives[changing_filter]
        
        if convert:
            representatives = representatives.replace(self.ecosystem.analyze.category_dict)

        return representatives


    @staticmethod
    def compare_clusters(clusters_df: pd.DataFrame, cluster_id1: str | int, cluster_id2: str | int) -> pd.DataFrame:
        """Compare qualitative values between two clusters.
        
        Returns a dataframe whose rows only display qualitative values that are different between 
        the clusters."""

        if isinstance(cluster_id1, int):
            cluster_id1 = 'c%d' % cluster_id1
        if isinstance(cluster_id2, int):
            cluster_id2 = 'c%d' % cluster_id2            
        
        comparative_df = clusters_df[[cluster_id1, cluster_id2]]
        
        # filter out rows where the two clusters share values
        changing_filter = comparative_df[cluster_id1] != comparative_df[cluster_id2]
        comparative_df = comparative_df[changing_filter]

        return comparative_df
    

    def plot_cluster_distribution(self, clusters_df: pd.DataFrame, cmap: str = 'Accent', figsize: tuple[int, int] = (10,5)) -> pd.DataFrame:
        """Plot the distribution of qualitative reaction categories per cluster.

        Each column in `clusters_df` is interpreted as a cluster and each row as a
        reaction category assignment. NaN values are treated as an additional
        'variable' category. Outputs a dataframe contaning the percentage of reactions per category for each cluster.
        """
        cluster_columns = list(clusters_df)
        n_reactions = clusters_df.shape[0]

        nan_rep = 100.0 # Change nan to additional category 'variable'
        filled_clusters_df = clusters_df.fillna(nan_rep)

        category_percents_dict = dict()

        for category in cluster_columns:
            vc = filled_clusters_df[category].value_counts()
            vc = 100 * vc/n_reactions # to percentages
            category_percents_dict[category] = vc.to_dict()   
            
        category_percents = pd.DataFrame.from_dict(category_percents_dict, orient='index')
        category_percents.fillna(0, inplace=True)
        category_percents.rename(columns = self.ecosystem.analyze.category_dict, inplace=True)

        # plot
        ax = category_percents.plot.barh(stacked=True, cmap=cmap, figsize=figsize)
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), title='reaction category')
        ax.xaxis.set_major_formatter(mtick.PercentFormatter())
        ax.set_ylabel('clusters')
        ax.set_xlabel('reactions')

        return category_percents 
    

# ======================================================= CLUSTER FUNCTIONS =======================================================


def get_hierarchical_clusters(dvector: np.ndarray, k: int = 20, lmethod: str = 'ward', 
                            criterion: str= 'maxclust', **kwards) -> tuple[int, np.ndarray]:
    row_linkage = hierarchy.linkage(dvector, method=lmethod)
    clusters = fcluster(row_linkage, k, criterion=criterion)

    k = len(np.unique(clusters))

    return k, clusters # clusters are indexed from 1


# retorna un vector de clusters 0-indexados
def get_DBSCAN_clusters(dmatrix: np.ndarray, eps: float = 0.05, 
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
    

def get_OPTICS_clusters(dmatrix: np.ndarray, max_eps: float = 0.05, 
                      min_samples: int = 5, **kwards) -> tuple[int, np.ndarray]:
    
    optics = OPTICS(max_eps = max_eps, min_samples = min_samples,metric='precomputed')
    clusters = optics.fit_predict(dmatrix)

    # output incluye outliers, valor -1 en vector clusters
    k = len(np.unique(clusters))

    return k, clusters # clusters are indexed from 0, includes outliers as -1


AssignLabels = Literal["kmeans", "discretize", "cluster_qr"]


def get_SC_clusters(dmatrix: np.ndarray, assign_labels: AssignLabels = "discretize",
                  random_state: int = 0, k: int = 20, delta: float = 0.2, 
                  **kwards) -> tuple[int, np.ndarray]:
    assert assign_labels in ['kmeans', 'discretize', 'cluster_qr']

    #transformación de matriz de distancia a matriz de similitud. Vía aplicación de Gaussian (RBF, heat) kernel:
    sim_matrix = np.exp(- dmatrix ** 2 / (2. * delta ** 2))
    
    sc = SpectralClustering(n_clusters = k, assign_labels = assign_labels, 
                            random_state = random_state, affinity = 'precomputed', **kwards)
    clusters = sc.fit_predict(sim_matrix)

    return k, clusters # clusters are indexed from 0


def get_AP_clusters(dmatrix: np.ndarray, **kwards) -> tuple[int, np.ndarray]:

    af = AffinityPropagation(**kwards)
    clusters = af.fit_predict(dmatrix) # np.ndarray

    # output incluye outliers, valor -1 en vector clusters
    k = len(np.unique(clusters))

    return k, clusters # clusters are indexed from 0, includes outliers as -1


