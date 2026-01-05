import numpy as np
import pandas as pd

import cobra
from cobra.util.array import create_stoichiometric_matrix
from scipy.sparse import lil_matrix

from benpy import vlpProblem
from benpy import solve as bensolve

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ecosystem_base import BaseEcosystem


class EcosystemCommunity():
    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem
        self.exchange_metabolite_info: dict[str, dict[str, Any]] = dict()
        self.mo_fba_sol = None
        self.non_blocked = None
        self.member_rxns = None
        
    @property
    def member_model_ids(self):
        return self.ecosystem.member_model_ids

    @property
    def community_model(self):
        return self.ecosystem.community_model


    # USADO EXPLICITAMENTE
    def set_pool_bounds(self, pool_metabolites, bioCons = None):
        """ 
        Changes pool exchange reaction bounds of metabolites in pool_metabolites.
        If a metabolite is not part of the pool a warning is raised and nothing is done.
        
        pool_metabolites: dictionary. 
                          keys: metabolite ids (without prefix, i.g., glyc_e) 
                          values: reaction bounds for the corresponding exchange reactions.
        factor: int. Factor to be considered in exchange reactions of the organisms of the community.
                    An organism can consume as maximum as value(from dict)*factor*fraction of that organism
                    in the community. Hence, bigger values make this constraint more flexible
        """
        
        for mid in pool_metabolites:
            if mid in self.exchange_metabolite_info:
                #Changing pool exchange reaction bounds
                rid = "EX_%s" % mid
                ex = self.community_model.reactions.get_by_id(rid)
                ex.bounds= pool_metabolites[mid]
                #Changing members exchange reaction bounds
                for member in self.exchange_metabolite_info[mid]:
                    ex_id = self.exchange_metabolite_info[mid][member]['ex_id']

                    if ex_id is not None:
                        ex = self.community_model.reactions.get_by_id(ex_id)
                        nlb = pool_metabolites[mid][0]
                        if nlb <= 0 and bioCons is not None:
                            #ex.lower_bound = nlb
                            ex.lower_bound = bioCons
                
            else:
                print("Skipping %s ..." % mid)
                print("Warning: %s is not a part of pool metabolites." % mid)

            self.non_blocked = None    


    def set_member_exchange_bounds(self, member_model_id, exchange_metabolites) -> None:
        """ 
        Changes member exchange reaction bounds of metabolites in exchange_metabolites.
        If a metabolite is not part of the exchanges a warning is raised and nothing is done.
        
        member_prefix: member model whose exchange reactions are modified
        exchange_metabolites: dictionary. 

        keys: metabolite ids (without prefix, i.g., glyc_e)
        
        values: reaction bounds for the corresponding exchange reactions.
        """            
            
        df =  self._get_exchange_df('ex_id') #id of member exchange reactions   
        
        for m in exchange_metabolites:
            new_bounds = exchange_metabolites[m]
            if m in df.index:
                rid = df.loc[m, member_model_id]
                if rid is not None:
                    rxn = self.community_model.reactions.get_by_id(rid)
                    rxn.bounds = new_bounds
                    self.exchange_metabolite_info[m][member_model_id]['bounds'] = new_bounds
                else:
                    print("No exchange reaction for %s in %s. Skypping..." % (m, member_model_id))
            else:
                print("No exchange or pool reactions for %s. Skypping" % m)
                 
                 
    def show_member_exchanges(self, mids=None):
        df = self._get_exchange_df('bounds')
        if mids is not None:
            df = df.loc[mids]

        return df    
    

    # done
    def _get_exchange_df(self, metabolite_attribute: str) -> pd.DataFrame: 
        '''Returns a dataframe that has the attribute speficied by 'metabolite_attribute',
        for each model and for each metabolite.'''

        assert metabolite_attribute in ['m_id', 'name', 'formula',' charge', 'ex_id', 'bounds']
    
        sorted_metabolite_indexes: list[str] = sorted(self.exchange_metabolite_info.keys()) 
        sorted_model_ids: list[str] = sorted(self.member_model_ids) 
        rows = list()

        for metabolite_index in sorted_metabolite_indexes:
            info = self.exchange_metabolite_info[metabolite_index]
            row = [info[model_id][metabolite_attribute] for model_id in sorted_model_ids]
            rows.append(row)
        
        df = pd.DataFrame(data=rows, index=sorted_metabolite_indexes, columns=sorted_model_ids)

        return df
    

    def _to_vlp(self, **kwargs):        
        """Returns a vlp problem from EcosystemModel."""
        # We are using bensolve-2.0.1:
        # B is coefficient matrix
        # P is objective Matrix
        # a is lower bounds for B
        # b is upper bounds for B
        # l is lower bounds of variables
        # s is upper bounds of variables
        # opt_dir is direction: 1 min, -1 max
        # Y,Z and c are part of cone definition. If empty => MOLP
        
        community_model = self.community_model
        Ssigma = create_stoichiometric_matrix(community_model, array_type="lil")
        
        vlp = vlpProblem(**kwargs)
        m, n = Ssigma.shape # mets, reactions
        q = self.ecosystem.size # number of members 
        vlp.B = Ssigma
        vlp.a = np.zeros((1, m))[0]
        vlp.b = np.zeros((1, m))[0]
        vlp.l = [r.lower_bound for r in community_model.reactions] 
        vlp.s = [r.upper_bound for r in community_model.reactions] 
        
        vlp.P = lil_matrix((q, n))
        vlp.opt_dir = -1
        
        for i, member_objectives in enumerate(self.ecosystem.objectives):
            for rid, coeff in member_objectives.items():
                rindex = community_model.reactions.index(rid)
                vlp.P[i,rindex] = coeff 
                
        vlp.Y = None
        vlp.Z = None
        vlp.c = None

        return vlp  
    

    def solve_mo_fba(self, bensolve_opts = None) -> None:
       
        if bensolve_opts is None:
            bensolve_opts = vlpProblem().default_options
            bensolve_opts['message_level'] = 0
        
        vlp_eco = self._to_vlp(options = bensolve_opts)    
        self.mo_fba_sol = bensolve(vlp_eco)


    def change_reaction_bounds(self, rid: str, new_bounds: tuple[int]) -> tuple[int]:
        community_model = self.community_model
        rxn = community_model.reactions.get_by_id(rid)
        old_bounds = rxn.bounds
        rxn.bounds = new_bounds
        return old_bounds

    
    def set_member_reactions(self):
        member_rxns = {x:[] for x in self.member_model_ids}
        for r in self.community_model.reactions:
            for member in self.member_model_ids:
                if r.id.startswith(member):
                    member_rxns[member].append(r.id)
                    break

        self.member_rxns = member_rxns         


    def _get_blocked_reactions(self):
        blocked = cobra.flux_analysis.find_blocked_reactions(self.community_model)
        return blocked     
        

    def _set_non_blocked_reactions(self) -> None:
        blocked = cobra.flux_analysis.find_blocked_reactions(self.community_model)
        all_ids = [x.id for x in self.community_model.reactions]
        non_blocked = set(all_ids).difference(set(blocked))
        self.non_blocked = non_blocked


    def _get_pareto_front(self) -> np.ndarray:
        #1. Front vertex:
        vv = self.mo_fba_sol.Primal.vertex_value[np.array(self.mo_fba_sol.Primal.vertex_type)==1]
        
        n_neg_vals = np.sum(vv<0)
        if n_neg_vals > 0:
            print('warning: Negative values in Pareto Front..')
            print(vv[vv<0])
            print("Changing negative values to zero...")
            vv[vv<0] = 0   

        return vv
    
