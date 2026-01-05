# local scripts
from eco_utils import *
from ecosystem_community import EcosystemCommunity
from ecosystem_grid import EcosystemGrid
from ecosystem_analyze import EcosystemAnalize
from ecosystem_plot import EcosystemPlot
from ecosystem_clustering import EcosystemClustering

from cobra import Model


class BaseEcosystem():
    def __init__(self, community_name: str = "community", community_id: str = "community"):
        self.community_name = community_name                
        self.community_id = community_id
                                 
        self.community_model: Model = Model(id_or_model=community_id, name=community_name)
        self.size: int  = 0   
        self.objectives: list[dict[str, float]] = [] 

        self.member_model_ids: list[str] = []
        
        self._build_modules()
        
        
    def _build_modules(self):
        self.community = EcosystemCommunity(self)
        self.grid = EcosystemGrid(self, points = None, feasible_points = None, pfractions = None, grid_step = None)
        self.analyze = EcosystemAnalize(self)
        self.plot = EcosystemPlot(self)
        self.clustering = EcosystemClustering(self)


    
