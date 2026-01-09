from collections import OrderedDict
from typing import Any, TYPE_CHECKING, cast, overload, Literal
from tqdm import tqdm
from joblib import Parallel, delayed

import numpy as np
import pandas as pd
import scipy.io as sio

from cobra.flux_analysis import flux_variability_analysis
from cobra.util.array import create_stoichiometric_matrix
from cobra.io.mat import _cell
from cobra import Reaction

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem


INFEASIBLE = -1000.0

CATEGORY_DICT = {
    -3.0: '-', 
    -2.0: '--',
    -1.0: '-0',
    0.0: '0',
    1.0: '0+',
    2.0: '++',
    3.0: '+',
    4.0: '-+',
    5.0: 'err',
    100.0: 'var'
    }


def qual_translate(fmin: np.ndarray, fmax: np.ndarray, delta: float = 1e-4) -> np.ndarray:
    """
    Translate FVA min/max values into qualitative states. Outputs the numeric value that maps
    to the qualitative state in `self.qualitative_dict`.
    """

    same_value = np.abs(fmax - fmin) < delta
    pos_max = fmax > delta
    neg_max = fmax < -delta
    pos_min = fmin > delta
    neg_min = fmin < -delta
    zero_max = np.abs(fmax) <= delta
    zero_min = np.abs(fmin) <= delta

    conditions = [
        neg_min & neg_max & same_value,
        neg_min & neg_max,
        neg_min & zero_max,
        zero_min & zero_max, 
        zero_min & pos_max,
        pos_min & pos_max & same_value,  # order here is VERY IMPORTANT, will fix later
        pos_min & pos_max, 
        neg_min & pos_max, 
    ]

    choices = [-3.0, -2.0, -1.0, 0.0, 1.0, 3.0, 2.0, 4.0]

    return np.select(conditions, choices, default=5.0)


class EcosystemAnalyze():
    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem  
        self.qual_vector_df: pd.DataFrame       = pd.DataFrame()    # public
        self.fva_reactions: list[str]           = []                # public
        self.fva_results: np.ndarray            = np.array([])
        self.qFCA = None
        self.category_dict: dict[float, str] = CATEGORY_DICT

        self.coupled_rxns: dict[str, dict]      = {}      
        self.member_blocked: dict[str, list]    = {}


    @property
    def member_model_ids(self) -> list[str]:
        return self.ecosystem.member_model_ids


    # ================================================== QUALITATIVE ANALYSIS ==================================================

    
    def select_reactions_for_fva(self) -> None:
        """Selects the set of community reactions to be used for FVA and clustering analyses.

        This method determines which reactions of the community model should be
        considered for analysis based on the availability of FCA results for all
        member models.

        If FCA results are missing for at least one member model, all non-blocked community 
        reactions are selected.

        If FCA results are available for all member models:
        - Blocked reactions in any member are excluded.
        - For each fully coupled reaction set, only one representative reaction is selected.
        - Reactions not accounted for by member models (e.g. pool or exchange reactions) are also included.
        
        Notes
        -----
        - This method assumes that FCA results, when available, have been previously
        stored using `store_fca_results`.
        - Pool bounds and community structure must be set before calling this method.
        - The method does not modify the community model; it only selects a subset
        of reactions for analysis.
        """
        community = self.ecosystem.community
        missing_models = set(self.member_model_ids) - set(self.coupled_rxns.keys())
    
        if missing_models: # FCA results are incomplete. Non-blocked reactions are obtained and ALL those reactions are used.        
            print(f"Missing FCA results for: {missing_models}.\nUsing non-blocked reactions only.")
            community._set_non_blocked_reactions()  

            fva_reactions = list(community.non_blocked)  
            fva_reactions.sort()
            self.fva_reactions = fva_reactions

            print("Total reactions considered for fva and clustering: %d" % len(self.fva_reactions))
            return

        # Otherwise, FCA results are available for all members, reactions for fva and clustering are reduced accordingly.
        accounted = []      
        coupled_rep = []
              
        # blocked reactions are not considered for fva and clustering   
        for blocked in self.member_blocked.values():
            accounted += blocked
              
        # Only one reaction from each fully coupled set is used for fva and clustering      
        for coupled_sets_dict in self.coupled_rxns.values():
            # reaction representative for coupled set:   
            coupled_rep += list(coupled_sets_dict.keys())
            coupled_sets = list(coupled_sets_dict.values())
            accounted += [reaction for cset in coupled_sets for reaction in cset]
        
        # Reactions for fva and clustering are those representing coupled sets and those not in any member (pool reactions)                   
        all_reaction_ids = [reaction.id for reaction in self.ecosystem.community_model.reactions]
        missing_reactions = set(all_reaction_ids).difference(set(accounted))
        
        fva_reactions = list(missing_reactions) + coupled_rep       
        fva_reactions.sort()
        self.fva_reactions = fva_reactions

        print("Total reactions considered for fva and clustering: %d" % len(self.fva_reactions))


    def analyze_grid(self, analysis: str = 'feasibility', update_bounds: bool = True, **kwargs) -> None:
        """
        Run an analysis over the full ecosystem grid.

        Updates attributes of `self.ecosystem.grid` and/or:
            - self.ecosystem.grid.feasible_points
            - self.qual_vector_df
            - self.fva_results

        Parameters
        ----------
        analysis : {'feasibility', 'qual_fva'}, default='feasibility'
            Type of analysis to run on full grid.
            - 'feasibility': checks if each grid point is feasible considering member fractions.
            - 'qual_fva'   : computes qualitative flux vectors and FVA results. If feasible 
            points are stored, analysis is run on those points only. FVA results are also stored.  

        update_bounds : bool, default=True
            Whether reaction bounds should be updated using member fractions
            at each grid point.
        """
        # calculate member fractions for each grid point if they are not stored
        if self.ecosystem.grid.member_fractions.size == 0:
            self.ecosystem.grid.set_member_fractions()
        
        # run analysis
        if analysis == 'feasibility':
            self._feasibility_analysis(update_bounds=update_bounds, **kwargs)
            
        elif analysis == 'qual_fva':
            self._qualitative_analysys(update_bounds=update_bounds, **kwargs)   

        else:
            raise ValueError(f"Non valid analysis option: {analysis}") 

    
    def _feasibility_analysis(self, update_bounds: bool = True, **kwargs) -> None:
        """Run feasibility analysis for all grid points.

        Stores a boolean grid, where position `i` is True if point `i` is feasible. 
        
        Parameters
        ----------
        update_bounds: bool, default True
            If True, update reaction bounds considering member community fractions before analysis.
        """
        points           = self.ecosystem.grid.points               
        member_fractions = self.ecosystem.grid.member_fractions
        n_points         = points.shape[0]
        n_frac           = member_fractions.shape[0]

        #print(f"point test: {points}, shape: {points.shape}")

        if update_bounds:
            if member_fractions.size == 0 or n_points != n_frac:
                raise RuntimeError("Missing or incomplete member fractions array. Cannot update reaction bounds!") 
            iterator = zip(points, member_fractions)
        else:
            iterator = [(points[i], None) for i in range(n_points)]

        # compute feasible points in parallel
        feasible = [self._analyze_point(p, f, analysis='feasibility',  update_bounds=update_bounds, **kwargs) 
                    for p, f in tqdm(iterator, total = n_points)]
    
        self.ecosystem.grid.feasible_points = np.asarray(feasible, dtype=bool)
        
        n_feasible = self.ecosystem.grid.feasible_points.sum()
        print(f"grid feasible points: {n_feasible}/{n_points}")


    def _qualitative_analysys(self, update_bounds: bool = True, **kwargs) -> None:
        """Run qualitative FVA analysis for all (or feasible) grid points.

        Generates qualitative vectors and FVA results for each grid point and stores them
        in `self.qual_vector_df` and `self.fva_results`. If feasible points have been
        calculated, the analysis is restricted to them, otherwise the full grid is used.

        - qual_vector_df : pd.DataFrame
            Dataframe containing qualitative categories assigned to each reaction.
        - fva_results : np.ndarray, shape (n_reactions, 2)
            Minimum and maximum flux values obtained from FVA.
        """
        points           = self.ecosystem.grid.points               
        member_fractions = self.ecosystem.grid.member_fractions
        feasible_points  = self.ecosystem.grid.feasible_points

        if feasible_points.size == 0:
            print("Warning: Feasible points have not been calculated. Running qualitative fva over full grid")
            df_index = np.arange(points.shape[0])
        else:
            print("Running qualitative fva over grid feasible points...")
            points = points[feasible_points, :]    
            member_fractions = member_fractions[feasible_points, :]     
            df_index = np.where(feasible_points)[0]
        
        fva_tuples = self._calculate_qual_vectors(points, member_fractions, update_bounds=update_bounds, **kwargs)
            
        qual_vector_list, fva_results = map(list, zip(*fva_tuples))    
        self.qual_vector_df = pd.DataFrame(np.array(qual_vector_list), columns=self.fva_reactions, index=df_index)
            
        fva_results = np.dstack(fva_results)
        fva_results = np.rollaxis(fva_results, -1)
            
        self.fva_results = fva_results  


    def _calculate_qual_vectors(self, grid_points: np.ndarray, member_fractions: np.ndarray, 
                                update_bounds: bool = True, **kwargs) -> list[tuple]:
        """Calculate qualitative FVA vectors for a set of grid points.

        Iterates over grid points and calculates qualitative FVA vectors using. Each element in the returned 
        list is a tuple `(qual_vector, fva_result)` for a point.
        """
        # Check for reactions selected for FVA and clustering
        if not self.fva_reactions:
            print("No reactions previously selected for FVA and clustering!\nSetting reactions for analysis...")
            self.select_reactions_for_fva()        
        
        n_points = grid_points.shape[0]
        n_frac   = member_fractions.shape[0]

        if update_bounds:
            if member_fractions.size == 0 or n_points != n_frac:
                raise RuntimeError("Missing or incomplete member fractions array. Cannot update reaction bounds!") 
            iterator = zip(grid_points, member_fractions)
        else:
            print("Warning: Calculating qualitative vectors without updating reaction bounds!")
            iterator = ((point, None) for point in grid_points)

        fva_tuples = [self._analyze_point(p, f, analysis='qual_fva', update_bounds=update_bounds, **kwargs) 
                      for p, f in tqdm(iterator, total = n_points)] 

        return fva_tuples
    

    @overload # feasibility analysis must return a bool
    def _analyze_point(self, grid_point: np.ndarray, member_fractions: np.ndarray | None, 
                       analysis: Literal["feasibility"], update_bounds: bool, delta: float) -> bool: ...


    @overload # fva analysis must return a tuple
    def _analyze_point(self, grid_point: np.ndarray, member_fractions: np.ndarray | None, 
                       analysis: Literal["qual_fva"], update_bounds: bool, delta: float) -> tuple: ...


    def _analyze_point(self, grid_point: np.ndarray, member_fractions: np.ndarray | None, 
                       analysis: str = 'feasibility', update_bounds: bool = False, delta: float = 1e-9) -> bool | tuple:
        """Analyze a single grid point of the ecosystem parameter space.

        This method evaluates either the feasibility of a grid point or computes
        qualitative flux variability information, optionally updating reaction
        bounds according to community member fractions.

        Parameters
        ----------
        grid_point : np.ndarray, shape (2,)
            Grid coordinates defining the community state. The first coordinate corresponds to the 
            fraction of total biomass growth assigned to member 1, and the second coordinate corresponds 
            to the total community biomass production rate. 

        member_fractions : np.ndarray or None, shape (n_members,)
            Relative abundance of each community member. These fractions are used to scale reaction 
            bounds when `update_bounds=True`. Gets ignored if `update_bounds=False`.

        analysis : {"feasibility", "qual_fva"}, default="feasibility"
            Type of analysis to perform:
            - "feasibility": checks whether the grid point admits a feasible solution.
            - "qual_fva": computes qualitative flux variability categories for a selected set of reactions.

        update_bounds : bool, default=False
            If True, reaction bounds are scaled according to member fractions before running the analysis. 

        delta : float, default=1e-9
            Numerical tolerance used when translating flux variability ranges into
            qualitative categories. Only relevant for `analysis="qual_fva"`.

        Returns
        -------
        bool
            If `analysis="feasibility"`, returns True if the grid point is feasible, False otherwise.

        tuple
            If `analysis="qual_fva"`, returns a tuple `(qual_vector, fva_values)`.
            
        Notes
        -----
        All model mutations (reaction bound updates and objective fixing) are performed inside a 
        `with community_model:` context manager. This guarantees that all changes are reverted after the 
        analysis is completed.
        """
        community_model = self.ecosystem.community_model

        #print(f"point: {grid_point}")
        fraction, mu_total = grid_point
        member_mu = np.array([fraction*mu_total, (1-fraction)*mu_total]) 

        with community_model:
            # update member reactions bounds if required:
            if update_bounds: 
                if not isinstance(member_fractions, np.ndarray):
                    raise TypeError("member_fractions must be a numpy array")
                
                #print('updating reaction bounds ...')    
                self.ecosystem.community.apply_member_fraction_bounds(community_model, member_fractions)


            # fix member objectives to grid point value:
            self.ecosystem.community.fix_growth_rates(community_model, member_mu)


            if analysis == 'feasibility': 
                # slim_optimize returns `error_value` if the model has no feasible solution.
                max_value = community_model.slim_optimize(error_value = INFEASIBLE)  

                if max_value != INFEASIBLE:
                    return True
                
                #print('unfeasible point')
                return False
            

            elif analysis == 'qual_fva':  # here we assume the point is feasible      
                if not self.fva_reactions:
                    raise RuntimeError('No reactions selected for fva and clustering!')
                    
                #print(f"running FVA on grid point: {grid_point}")
                
                rxn_fva = flux_variability_analysis(community_model, reaction_list=self.fva_reactions) # type: ignore              
                rxn_fva = rxn_fva.loc[self.fva_reactions, :] # just to make sure reactions are in the 
                                                             # same order as fva_reactions
                minimum_values = rxn_fva["minimum"].to_numpy()
                maximum_values = rxn_fva["maximum"].to_numpy()

                #print("translating to qualitative vector..")
                qualitative_vector = qual_translate(minimum_values, maximum_values, delta=delta)
                fva_results = rxn_fva.values

                return list(qualitative_vector), fva_results


            else:
                raise ValueError(f"Non valid analysis option: {analysis}")     


    # ================================================== QUANTITATIVE GRID ANALYSIS ==================================================


    def quan_FCA(self, grid_x: list[float], grid_y: list[float], reaction_ids: list[str]) -> None:
        """Performs quantitative Flux Coupling Analysis on two reactions (rxns_analysis) on points of a sub-grid defined by points grid_x, grid_y.
        
        Stores an attribute self.qFCA containing a dataframe with the following columns: 
            flux_rxns_analysis[0], flux_rxn_analysis[1], FVA (str: minimum or maximum), point (coordinates of point)
        """
        assert len(reaction_ids) == 2

        feasible_points = self.ecosystem.grid.points[self.ecosystem.grid.feasible_points]
        reaction_id_0 = reaction_ids[0]
        reaction_id_1 = reaction_ids[1]

        print('Quantitative Flux Coupling analysis \n Initializing grid...')

        analyze_points = []
        # Match points defined by the user in grid_x, grid_y to specific points on the grid
        for y in grid_y:
            for x in grid_x:
                search_point = np.array([x, y])
                distances = np.linalg.norm(feasible_points-search_point, axis=1)
                min_index = np.argmin(distances)
                analyze_points.append(min_index)
                print(f"The closest point to {search_point} is {feasible_points[min_index]}, at a distance of {distances[min_index]}")

        qFCA_data = []

        for point in analyze_points:
            fraction, mu_total = feasible_points[point]
            fraction, mu_total = float(fraction), float(mu_total)
            
            member_fractions = np.array([fraction, 1-fraction])
            mu_array = np.array([fraction*mu_total, (1-fraction)*mu_total]) # equivalent to old grid
            
            print(f'Selected point: {fraction, mu_total}')
            print(f'This point frac {member_fractions}')
            print(f'Old grid point {mu_array}')

            with self.ecosystem.community_model as community_model:
                # update bounds nad objectives
                self.ecosystem.community.apply_member_fraction_bounds(community_model, member_fractions)
                self.ecosystem.community.fix_growth_rates(community_model, mu_array)

                # define limit reactions based on theoretical max-min defined from model
                fva_result = flux_variability_analysis(community_model, reaction_list = [reaction_id_0])
                min_value = float(fva_result['minimum'].iloc[0])
                max_value = float(fva_result['maximum'].iloc[0])
                values_rxn_ref = np.linspace(min_value, max_value, num=50)

                norm_0, norm_1 = self._fraction_to_normalize(member_fractions, reaction_ids)
                reaction_0 = cast(Reaction, community_model.reactions.get_by_id(reaction_id_0))
                
                for value in values_rxn_ref:
                    reaction_0.bounds = (value, value)
                    fva_result = flux_variability_analysis(community_model, reaction_list = [reaction_id_1])
                    
                    for bound in fva_result: # [minimum, maximum]
                        qFCA_data.append({
                            reaction_id_0: value/norm_0,
                            reaction_id_1: fva_result[bound].iloc[0]/norm_1,
                            'FVA': bound,
                            'point': f"{fraction:.3f}, {mu_total:.3f}"
                        })

        self.qFCA = pd.DataFrame(qFCA_data)
        
        
    def _fraction_to_normalize(self, member_fractions: np.ndarray, reaction_ids: list[str]) -> tuple[float, float]:
        #from point_fraction computes which element of this array should be used for normalization
        #reaction: string reaction id
        fractions = set()

        for reaction_id in reaction_ids:
            for i, model_id in enumerate(self.member_model_ids):
                if reaction_id.startswith(model_id+'_'):
                    fractions.add(member_fractions[i])
                    break
            else:
                fractions.add(1.0)
        
        return tuple(fractions)


    def write_fca_input(self,model_id,file_dir, discard_zero_bound=True):
     
        file_name = "%s/%s_fca_input.mat" % (file_dir,model_id)
        community_model = self.ecosystem.community_model 
            
        rxns = community_model.reactions
        mets = community_model.metabolites
        stoich_mat = create_stoichiometric_matrix(community_model)
        rids = np.array(rxns.list_attr("id"))
        mids = np.array(mets.list_attr("id"))
        rev  = np.array(rxns.list_attr("reversibility"))*1
    
        #discard reactions from pool and other members and also reactions from prefix with zero lower and upper bounds 
        #these last reactions are added to blocked.
        
        to_discard = []
        blocked = [] 
                
        for ix in range(len(rids)):
            rid = rids[ix]
            if not rid.startswith(model_id):
                to_discard.append(ix)
            else:        
                if discard_zero_bound:            
                    r = cast(Reaction, community_model.reactions.get_by_id(rid))
                    if r.lower_bound == 0 and r.upper_bound == 0 :
                        to_discard.append(ix)
                        blocked.append(rid)     
        
        if len(to_discard)>0:
            rids = np.delete(rids,to_discard)
            rev = np.delete(rev,to_discard)
            stoich_mat = np.delete(stoich_mat, to_discard, axis=1)
                  
        
        #discard metabolites from pool and other members:
        to_discard = []
        for ix in range(len(mids)):
            mid=mids[ix]
            if not mid.startswith(model_id):
                to_discard.append(ix)
        
        if len(to_discard)>0:  
            mids = np.delete(mids,to_discard)
            stoich_mat = np.delete(stoich_mat, to_discard, axis=0)    
        
        rids =list(rids)
        mids =list(mids)
        
        #create mat objects for FCA and stored them in an output file 
        mat  = OrderedDict()    
        mat["Metabolites"] = _cell(mids)
        mat["Reactions"] = _cell(rids)
        mat["stoichiometricMatrix"] = stoich_mat
        mat["reversibilityVector"] = rev 
        
        varname1 = 'fca_input'
        varname2 = 'bound_blocked'
        #varname2 = "%s_bound_blocked" % prefix
        #sio.savemat(file_name, {varname1: mat, varname2:mat2}, oned_as="column")
        sio.savemat(file_name, {varname1: mat, varname2:_cell(blocked)}, oned_as="column")
        print("%s FCA's input in %s" % (model_id, file_name))
        print("   stoichiometric matrix : %s" % str(stoich_mat.shape))
        

    def store_fca_results(self,model_id,fca_file):

        mat_contents = sio.loadmat(fca_file)
        fctable = mat_contents['fctable']
        blocked = mat_contents['blocked'][0]
        rxn_ids = np.array([rid[0][0] for rid in mat_contents['rxns']])
        bound_blocked = np.array([rid[0][0] for rid in mat_contents['bound_blocked']])
        blocked_ids = rxn_ids[blocked==1]
        blocked_ids = list(blocked_ids) + list(bound_blocked)
        non_blocked_ids = list(rxn_ids[blocked!=1])
        
        g = np.unique(fctable==1,axis=0)
        df = pd.DataFrame(data=g, columns= non_blocked_ids)
        coupled_sets=dict() #key = id of one reaction of the coupled set (first in alpha order); 
                            #value = list of coupled rxns ids
        for x in df.index:
            coupled = list(df.columns[df.loc[x]])
            coupled.sort()
            coupled_sets[coupled[0]] = coupled   
                
        self.coupled_rxns[model_id] = coupled_sets
        self.member_blocked[model_id]= blocked_ids
        total_rxns = len(rxn_ids)+len(bound_blocked)
        print("Flux coupling results for member %s stored:" % model_id)
        print("   Total reactions: %d" % total_rxns)
        print("   Fully coupled reaction sets: %d" % len(coupled_sets))
        print("   Blocked reactions: %d" % len(blocked_ids))      
        print("-")
