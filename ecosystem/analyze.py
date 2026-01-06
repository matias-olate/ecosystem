import numpy as np
import pandas as pd

from cobra.flux_analysis import flux_variability_analysis
from cobra.util.array import create_stoichiometric_matrix
from cobra.io.mat import _cell

import copy
import scipy.io as sio
from collections import OrderedDict
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ecosystem.base import BaseEcosystem


class EcosystemAnalize():
    def __init__(self, base_ecosystem: "BaseEcosystem"):
        self.ecosystem = base_ecosystem  # parent class
        self.qual_vector_df = None # public
        self.rxn2cluster = None
        self.fva_results = None
        self.qFCA = None

        self.coupled_rxns    = {model_id: None for model_id in self.ecosystem.member_model_ids}      
        self.member_blocked  = {model_id: None for model_id in self.ecosystem.member_model_ids}

    @property
    def member_model_ids(self):
        return self.ecosystem.member_model_ids


    # CUALITATIVE ANALYSIS

    
    def set_cluster_reactions(self):
        community = self.ecosystem.community

        # if FCA has been performed and results stored for all members, reactions for fva and clustering are reduced accordingly.
        # Otherwise, non-blocked reactions are obtained and ALL those reactions are used.       
        coupled_dicts =  list(self.coupled_rxns.values())
              
        if coupled_dicts.count(None) != 0: # at least one member without FCA results       
            print("Missing FCA results")
            print("Using non-blocked reactions only")
            community._set_non_blocked_reactions()  
            rxn2cluster = list(community.non_blocked)  
            #rxn2cluster = [r.id for r in self.cmodel.reactions]

        else:      
            accounted = []      
            coupled_rep = []
              
            # blocked reactions are not considered for fva and clustering   
            for model_id in self.member_blocked:
                blocked = self.member_blocked[model_id]
                accounted = accounted + blocked
              
            # Only one reaction from each fully coupled set is used for fva and clustering      
            for model_id in self.coupled_rxns:
                coupled_sets_dict = self.coupled_rxns[model_id]
                # rxn representative for coupled set:   
                coupled_rep  = coupled_rep + list(coupled_sets_dict.keys())
                coupled_sets = list(coupled_sets_dict.values())
                accounted = accounted + [rxn for cset in coupled_sets for rxn in cset]
        
            # Reactions for fva and clustering are those representing coupled sets and those not in any member (pool reactions)                   
            all_rids = [r.id for r in self.ecosystem.community_model.reactions]
            missing = set(all_rids).difference(set(accounted))
        
            rxn2cluster = list(missing) + coupled_rep       
        
        rxn2cluster.sort()
        self.rxn2cluster =  rxn2cluster
        print("Total reactions considered for fva and clustering: %d" % len(self.rxn2cluster))


    def calculate_qual_vectors(self,point_array, pfraction_array=None, update_bounds=False, **kwargs):
        
        # Check for reactions selected for FVA and clustering
        if self.rxn2cluster is None:
            print("No reactions previously selected for FVA and clustering!\nSetting reactions to cluster...")
            self.set_cluster_reactions()        
        
        if not update_bounds:
            print("Warning:Calculating qualitative vectors without updating reaction bounds!")
            r = [self._analyze_point((p,None), analysis = 'qual_fva', update_bounds=False, **kwargs) for p in point_array] 
        else:
            npoints = point_array.shape[0]
            nfrac   = pfraction_array.shape[0]
            
            if pfraction_array is None or npoints != nfrac:
                raise RuntimeError("Missing or incomplete member fractions array. Cannot update rxn bounds!!") 
            else:
                coord_frac = [(point_array[ix],pfraction_array[ix]) for ix in range(npoints)]  
                #list of tuples
                r = [self._analyze_point(x, analysis = 'qual_fva', update_bounds = True, **kwargs) for x in coord_frac] 
         
        return r
    

    def check_feasible(self, point_array, pfraction_array=None, update_bounds=False, **kwargs):
        if not update_bounds:
            r = [self._analyze_point((p,None), analysis = 'feasibility', update_bounds=False, **kwargs) for p in point_array] 
            
        else:
            npoints = point_array.shape[0]
            nfrac   = pfraction_array.shape[0]
            if pfraction_array is None or npoints != nfrac:
                raise RuntimeError("Missing or incomplete member fractions array. Cannot update rxn bounds!!") 
            else:
                coord_frac = [(point_array[ix],pfraction_array[ix]) for ix in range(npoints)]    
                r = [self._analyze_point(x, analysis = 'feasibility', update_bounds = True, **kwargs) for x in coord_frac] 
            
        return r


    def _analyze_point(self, ptuple, analysis = 'feasibility', update_bounds = False, delta = 1e-9):
        
        # ptuple: tuple of two arrays:   (0) Grid point coordinates; 
        #                                (1) grid point member fractions; grid point
        # analysis: type of analysis to run. Options:
        #           'feasibility': check if grid point is feasible
        #           'qual_fva'   : get grid point vector of rxn qualitative values. 
        #                           
                    
        # update_bounds: if True update reaction bounds considering member community fractions 
        #                before analysis 
        # delta: threshold value to consider flux differences as zero when comparing fva min and max values
        #        ('qual_fva' option only)  
        # Returns  
        #  boolean indicating point feasibility ('feasibility' analysis) or  ('qual_val' analysis)
        #  a tuple where the first element is a list of rxns qualitative values for the analyzed grid point
        #  and the second is an array with the corresponding fva results
        def qual_translate(x, delta = 1e-4):
            fvaMax = x.maximum
            fvaMin = x.minimum
           
            #3 fixed on positive value
             #-3 fixed on negative value
             
            # -2: max and min <0
            if fvaMax < -delta and fvaMin < -delta:
                ret = -2
                if abs(fvaMax-fvaMin) < delta:
                    ret = -3
            # 0: max and min == 0
            elif (fvaMax >= -delta and fvaMax <=delta) and (fvaMin >=-delta and fvaMin <=delta):
                ret = 0
            # -1: max and min <=0
            elif (fvaMax>= -delta and fvaMax < delta) and (fvaMin < -delta):
                ret = -1
            # 1: max and min >= 0
            elif (fvaMin >=-delta and fvaMin <=delta) and fvaMax >delta:
                ret = 1
            # 2: max and min >0
            elif (fvaMax >delta and fvaMin > delta):
                ret = 2
                if abs(fvaMax - fvaMin) < delta:
                    ret = 3
            elif (fvaMin < -delta and fvaMax > delta):
                ret = 4 #reversible
            else:
                ret = 5 #nans            
         
            return ret          
        
        # 0,0+,0-,++,--,+,-,+-,*
        
        community_model = self.ecosystem.community_model
        community = self.ecosystem.community

        point = ptuple[0] #grid point coordinates
        print(point)
        point = [point[0]*point[1], (1-point[0])*point[1]]#equivalent of old point 
        pfrac = ptuple[1] #grid point member fraction
            
        out = None
        
        with community_model:
           # update member reactions bounds if required:
           if update_bounds: 
                print('updating reaction bounds ...')
                # In the case that all objectives are zero, all member fractions are nan
                # Here we set fractions to zero to avoid errors setting bounds
                if np.all(np.isnan(pfrac)): 
                    pfrac = np.array([0.0]*pfrac.size)                
                    
                # reactions are assign to each community member
                community.set_member_reactions()
                for i, member in enumerate(self.member_model_ids):
                    mfrac = pfrac[i]
                    mrxns = community.member_rxns[member]
           
                    # rxn bounds are updated, accounting for members fractions in the community 
                    
                    for rid in mrxns:
                        r = community_model.reactions.get_by_id(rid)
                        old_bounds = r.bounds
                        r.bounds =(old_bounds[0] * mfrac, old_bounds[1] * mfrac)

           # fix member objectives to grid point value:
           for ix, member_objectives in enumerate(self.ecosystem.objectives):    
                if len(member_objectives) != 1:
                    raise RuntimeError("Warning: More than one reaction in %s objective function. Not supported!!" 
                      % self.member_model_ids[ix])        
                #new bounds for member ix objective function reaction:
                new_bounds = (point[ix],point[ix])    
                #newGrid NJ
                #print(point[ix])

            
                #change bounds for each objective reaction
                for rid in member_objectives.keys(): # member_objectives should be single key dictionary
                    rxn = community_model.reactions.get_by_id(rid)
                    rxn.bounds = new_bounds

                    #set one of objective reactions as community objective
                    #cmodel.objective = rid #commented since the model comes with an objective
            
           
           if analysis == 'feasibility': 
                #check point feasibility
                error_value = -1000
                ob = community_model.slim_optimize(error_value = error_value)  
                if ob == error_value:
                    out = False
                    print('unfeasible point')
                else:
                    out = True
        
           elif analysis == 'qual_fva':  # here we assume the point is feasible      

                if self.rxn2cluster is None:
                    raise RuntimeError('No reactions selected for fva and clustering!')
                    
                print("running FVA on grid point...")
                print(ptuple)
                
                rxn_fva = flux_variability_analysis(community_model, reaction_list= self.rxn2cluster)
                
                rxn_fva = rxn_fva.loc[self.rxn2cluster,:] # just to make sure reactions are in the 
                                                         # same order as rxn2cluster
                    
                print("translating to qualitative vector..")
                out = (list(rxn_fva.apply(qual_translate, axis = 1, delta = delta)), #qual_vector 
                             rxn_fva.values) # array with FVA results  
                
        return out      
        

    def analyze_grid(self, analysis = 'feasibility', update_bounds=True, **kwargs):
        #analysis: type of analysis to run on full grid:
        #   Options:         
        #       feasibility = checks if each grid point is feasible considering member fractions
        #       qual_fva = calculates qualitative vectors for each point in the grid. If feasible 
        #                  points are stored, analysis is run on those points only.   
        #                  FVA results are also stored.   
        
        #step 1: # calculate community distribution (member fractions) for each grid point 
                 # if they are not stored
        if self.ecosystem.grid.pfractions is None:
            self.ecosystem.grid.set_points_distribution()
        
        point_array     = self.ecosystem.grid.points               
        pfraction_array = self.ecosystem.grid.pfractions
        
        #Option 'feasibility':
        if analysis == 'feasibility':
                
            #step 2: run feasibility analysis for all grid points
            npoints = point_array.shape[0]       
            self.ecosystem.grid.feasible_points = self.check_feasible(point_array, pfraction_array, 
                                                       update_bounds=update_bounds, **kwargs)
            
            npoints = point_array.shape[0]
            nfeasible = sum(self.ecosystem.grid.feasible_points)
            print("grid feasible points: %d / %d" % (nfeasible, npoints))
            
        elif analysis == 'qual_fva':
                
            #step 2: check for previously calculated feasible points  
            feasible_points = self.ecosystem.grid.feasible_points
            if feasible_points is None:
                print("Warning: Feasible points have not been calculated. Running qualitative fva over full grid")
                df_index = np.arange(point_array.shape[0])
            else:
                print("Running qualitative fva over grid feasible points...")
                point_array = point_array[feasible_points,:]    
                pfraction_array = pfraction_array[feasible_points,:]     
                df_index =  np.where(feasible_points)[0]
        
            rtuples = self.calculate_qual_vectors(point_array,pfraction_array,update_bounds=update_bounds, **kwargs)
            
            qual_vector_list, fva_results =  map(list, zip(*rtuples))    
            self.qual_vector_df = pd.DataFrame(np.array(qual_vector_list),columns = self.rxn2cluster, index=df_index)
            
            fva_results = np.dstack(fva_results)
            fva_results = np.rollaxis(fva_results,-1)
            
            self.fva_results = fva_results  


    # QUANTITATIVE GRID ANALYSIS


    def quan_FCA(self, grid_x, grid_y, rxns_analysis):
        #Performs quantitative Flux Coupling Analysis on two reactions (rxns_analysis) and on points of a sub-grid defined by points grid_x, grid_y
        #returns: a dataframe with columns to plot afterwards
        #Columns: flux_rxns_analysis[0], flux_rxn_analysis[1], FVA (str: minimum or maximum), point (coordinates of point)

        feasible_points = self.ecosystem.grid.points[self.ecosystem.grid.feasible_points]
        analyze_points = []
        print('Quantitative Flux Coupling analysis \n Initializing grid...')


        # a lo mejor conviene definirlo fuera
        def fraction_to_normalize(point_fractions, reaction):
            #from point_fraction computes which element of this array should be used for normalization
            #reaction: string reaction id
            fraction = ''
            for i, pre in enumerate(self.member_model_ids):
                if reaction.startswith(pre+'_'):
                    fraction = point_fractions[i]

            
            if fraction=='':
                print('No org detected, asumming community reaction')
                fraction =1
        
            return(fraction)
        
        #Match points defined by the user in grid_x, grid_y to specific points on the grid
        for y in grid_y:
            for x in grid_x:
                search_point = [x, y]
                distances = np.linalg.norm(feasible_points-search_point, axis=1)
                min_index = np.argmin(distances)
                analyze_points.append(min_index)
                print(f"the closest point to {search_point} is {feasible_points[min_index]}, at a distance of {distances[min_index]}")


        maxmin_data = []
        for this_point in analyze_points:
            eco = self.ecosystem
            community_model = copy.deepcopy(eco.community_model)
        
            this_point_coords = feasible_points[this_point]
            print('Selected point'+str(this_point_coords))
            print('This point coords '+str(this_point_coords))
            this_point_frac = [this_point_coords[0], 1-this_point_coords[0]]
            print('This point frac '+str(this_point_frac))
            point = [this_point_coords[0]*this_point_coords[1], (1-this_point_coords[0])*this_point_coords[1]] #equivalent to old grid
            print('Old grid point '+str(point))

            #update bounds
            for i, member in enumerate(self.member_model_ids):
                mfrac = this_point_frac[i]
                mrxns = eco.community.member_rxns[member]

                for rid in mrxns:
                    r = community_model.reactions.get_by_id(rid)
                    old_bounds = r.bounds
                    r.bounds = (old_bounds[0]*mfrac, old_bounds[1]*mfrac)

            for ix, member_objectives in enumerate(eco.objectives):
                new_bounds = (point[ix], point[ix])

                for rid in member_objectives.keys():
                    rxn = community_model.reactions.get_by_id(rid)
                    rxn.bounds = new_bounds

            #try:
            #define limits reactions based on theoretical max-min defined from model
            rxn_ref_fva = flux_variability_analysis(community_model, reaction_list = rxns_analysis[0])

            #define range reactions
            values_rxn_ref = np.linspace(rxn_ref_fva['minimum'].iloc[0], rxn_ref_fva['maximum'].iloc[0], num=50)

            with community_model:
                for val in values_rxn_ref:
                    rxn = community_model.reactions.get_by_id(rxns_analysis[0])
                    rxn.bounds = (val,val)
                    #compute max min
                    fva = flux_variability_analysis(community_model, reaction_list = rxns_analysis[1])
                    for i, el in enumerate(fva):
                        row_dict = dict()
                        row_dict[rxns_analysis[0]] = val/fraction_to_normalize(this_point_frac, rxns_analysis[0])
                        row_dict[rxns_analysis[1]] = fva[el].iloc[0]/fraction_to_normalize(this_point_frac, rxns_analysis[1])
                        row_dict['FVA'] = el
                        row_dict['point'] = str([round(this_point_coords[0],3), round(this_point_coords[1],3)])
                        maxmin_data.append(row_dict)

            #except:
            #    print('\n Issues with '+str(this_point_coords)+' unfeasible?')
        
        self.qFCA = pd.DataFrame(maxmin_data)
        
        
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
                    r = community_model.reactions.get_by_id(rid)
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
