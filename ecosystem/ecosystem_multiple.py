import numpy as np
import pandas as pd
import pickle

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.ticker as mtick

import cobra
from cobra import Metabolite, Reaction, Model
from cobra.util.array import create_stoichiometric_matrix
from cobra.util.solver import linear_reaction_coefficients
from cobra.flux_analysis import flux_variability_analysis
from cobra.io.mat import _cell

from benpy import vlpProblem
from benpy import solve as bensolve

from scipy.sparse import lil_matrix
from scipy.spatial import Delaunay
from scipy.spatial import distance
import scipy.io as sio

from functools import reduce
from collections import OrderedDict

from eco_utils import *
from base_ecosystem import BaseEcosystem


from scipy.cluster.hierarchy import fcluster
from scipy.cluster import hierarchy


class Ecosystem(BaseEcosystem):

    def __init__(self, model_prefix, community_name = 'community', file_dir=''):
        super().__init__(community_name)
        
        pickle_filename = "%s.p" % model_prefix
        cobra_model_filename = "%s.json" %model_prefix
        data_dict = pickle.load( open(file_dir+pickle_filename, "rb" ) )
        

        self.size = 2 #hardcoded
        self.prefixes = data_dict['prefixes']
        self.coupled_rxns    = {prefix:None for prefix in self.prefixes}      
        self.member_blocked  = {prefix:None for prefix in self.prefixes}

        self.points = data_dict['points']
        self.objectives = data_dict['objectives']
        self.rxn2cluster = data_dict['rxn2cluster']
        self.oldRxn2Cluster = None


        self.member_rxns = data_dict['member_rxns']
        self.feasible_points = data_dict['feasible_points']
        
        self.pfractions = np.array([[p[0], 1-p[0]] for p in self.points]) #assuming two organisms
        
        self.k = None
      

    def set_member_exchange_bounds(self, member_prefix, exchange_metabolites):
        """ 
        Changes member exchange reaction bounds of metabolites in exchange_metabolites.
        If a metabolite is not part of the exchanges a warning is raised and nothing is done.
        
        member_prefix: member model whose exchange reactions are modified
        exchange_metabolites: dictionary. 
                          keys: metabolite ids (without prefix, i.g., glyc_e)
                          values: reaction bounds for the corresponding exchange reactions.
        """            
            
            
        df =  self.get_exchange_df('ex_id') #id of member exchange reactions   
        
        for m in exchange_metabolites:
            new_bounds = exchange_metabolites[m]
            if m in df.index:
                rid = df.loc[m, member_prefix]
                if rid is not None:
                    rxn = self.community_model.reactions.get_by_id(rid)
                    rxn.bounds = new_bounds
                    self.exchange_metabolite_info[m][member_prefix]['bounds'] = new_bounds
                else:
                    print("No exchange reaction for %s in %s. Skypping..." % (m, member_prefix))
            else:
                print("No exchange or pool reactions for %s. Skypping" % m)
                 
    def show_member_exchanges(self, mids=None):
        df = self.get_exchange_df('bounds')
        if mids is not None:
            df = df.loc[mids]
        return df    
    
    def get_exchange_df(self, k): 
    
        index = sorted(self.exchange_metabolite_info.keys())
        columns = sorted(self.prefixes)
        rows=list()
        for m in index:
            info = self.exchange_metabolite_info[m]
            row = [info[prefix][k] for prefix in columns]
            rows.append(row)
        
        df = pd.DataFrame(data=rows, index=index, columns=columns)
        return df
    
    def _to_vlp(self,**kwargs):        
        """Returns a vlp problem from EcosystemModel"""
        # We are using bensolve-2.0.1:
        # B is coefficient matrix
        # P is objective Matrix
        # a is lower bounds for B
        # b is upper bounds for B
        # l is lower bounds of variables
        # s is upper bounds of variables
        # opt_dir is direction: 1 min, -1 max
        # Y,Z and c are part of cone definition. If empty => MOLP
        
        cmodel = self.community_model
        Ssigma = create_stoichiometric_matrix(cmodel, array_type="lil")
        
        vlp = vlpProblem(**kwargs)
        m, n = Ssigma.shape # mets, reactions
        q = self.size # number of members 
        vlp.B = Ssigma
        vlp.a = np.zeros((1, m))[0]
        vlp.b = np.zeros((1, m))[0]
        vlp.l = [r.lower_bound for r in cmodel.reactions] 
        vlp.s = [r.upper_bound for r in cmodel.reactions] 
        
        
        vlp.P = lil_matrix((q, n))
        vlp.opt_dir = -1
        
        for i, member_objectives in enumerate(self.objectives):
            for rid, coeff in member_objectives.items():
                rindex = cmodel.reactions.index(rid)
                vlp.P[i,rindex] = coeff 
                
        vlp.Y = None
        vlp.Z = None
        vlp.c = None
        return vlp  
    
    def mo_fba(self, bensolve_opts = None):
       
        if bensolve_opts is None:
            bensolve_opts = vlpProblem().default_options
            bensolve_opts['message_level'] = 0
        
        vlp_eco = self._to_vlp(options = bensolve_opts)    
        self.mo_fba_sol = bensolve(vlp_eco)
         
    def get_polytope_vertex(self, expand=True):
  
        """
        polytope: pareto front + axes segments + extra segments perpendicular to axes dimensions where 
        pareto solutions don't reach 0 values. 
        (assumption: objective functions can only take positive values)
        """
        
        #1. Front vertex:
        vv = self.mo_fba_sol.Primal.vertex_value[np.array(self.mo_fba_sol.Primal.vertex_type)==1]
        
        n_neg_vals = np.sum(vv<0)
        if n_neg_vals > 0:
            print('warning: Negative values in Pareto Front..')
            print(vv[vv<0])
            print("Changing negative values to zero...")
            vv[vv<0] = 0        
        
        
        #2. origin
        ov = np.zeros((1,self.size))
        
        
        if expand == True:
            #3. Extra points that close polytope (only if points (0,0,0,...,xi_max,0,...0) are not pareto front 
            # points but they are feasible) 
            
            #MP: si hay que incluir estos puntos significa que hay miembros que son givers: i.e. pueden crecer
            # a su máxima tasa y aguantar que otros miembros crezcan también
            # si un punto (0,0,0,...,xi_max,0,...0) no es factible entonces el miembro i necesita que otros crezcan 
            # para poder crecer (needy).
            
            #3.1 Check if points  (0,0,0,...,xi_max,0,...0) are part of the pareto front
            n = self.size - 1
            all_zeros_but_one = np.argwhere(np.sum(vv == 0,axis=1)==n) # index of (0,0,...,xi_max,0,0...0) points
            all_zeros_but_one = all_zeros_but_one.flatten()
    
            # indexes i of non-zero member in (0,0,...,xi_max,0,0...0) pareto points, 
            # i.e. members that are not givers nor needy. 
            non_zero_dims =  np.argmax(vv[all_zeros_but_one,:], axis = 1) 
            
            # givers and/or needy members:
            givers_or_needy_indexes = np.setdiff1d(np.array(range(self.size)), non_zero_dims) 
            gn_total= len(givers_or_needy_indexes)    
        
            #3.2 Check if non-pareto points (0,0,0,...,xi_max,0,...0) are feasible 
            if gn_total >0:
                # max values for giver_or_needy members:
                max_vals = np.max(vv, axis=0)
                cpoints = np.diag(max_vals)
                to_check = cpoints[givers_or_needy_indexes,:]
                
                are_feasible = self.check_feasible(to_check)
                
                ev = to_check[are_feasible,:] 
                polytope_vertex = np.concatenate((vv,ov,ev), axis=0)
            else: 
                polytope_vertex = np.concatenate((vv,ov), axis=0)
        
        else:
                polytope_vertex = np.concatenate((vv,ov), axis=0)
        return polytope_vertex 
    
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
    
    def calculate_qual_vectors(self,point_array, pfraction_array=None, update_bounds=False, **kwargs):
        
        # Check for reactions selected for FVA and clustering
        if self.rxn2cluster is None:
            print("No reactions previously selected for FVA and clustering!")
            print("Setting reactions to cluster...")
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
    
    def _analyze_point(self, ptuple, analysis = 'feasibility', update_bounds = False, delta = 1e-4):
        
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

            
            ## category - (-3): max = min < 0
            #if (fvaMax - fvaMin) <= delta and fvaMax < -delta:
            #    ret = -3
            
            ## category + (5): max = min > 0
            #elif (fvaMax - fvaMin) <= delta and fvaMin > delta:
            #    ret = 5     
            
            ## category -- (-2): max and min <0     
            #elif fvaMax < -delta and fvaMin < -delta:
            #    ret = -2
            
            ## category 0  (0): max and min == 0
            #elif (fvaMax >= -delta and fvaMax <=delta) and (fvaMin >=-delta and fvaMin <=delta):
            #    ret = 0
            
            ## category -0 (-1): max and min <=0
            #elif (fvaMax>= -delta and fvaMax < delta) and (fvaMin < -delta):
            #    ret = -1
            
            ## category 0+ (1): max and min >= 0
            #elif (fvaMin >=-delta and fvaMin <=delta) and fvaMax >delta:
            #    ret = 1
            
            ## category ++ (2): max and min >0
            #elif (fvaMax >delta and fvaMin > delta):
            #    ret = 2
            
            ## category -+ (3): max > 0 and min < 0     
            #elif (fvaMin < -delta and fvaMax > delta): 
            #    ret = 3            
            
            ## category * (4): something weird happened!
            #else:
            #    ret = 4 #nans            
            return ret          
        
        # 0,0+,0-,++,--,+,-,+-,*
        
        cmodel = self.community_model
        point = ptuple[0] #grid point coordinates
        print(point)
        point = [point[0]*point[1], (1-point[0])*point[1]]#equivalent of old point 
        pfrac = ptuple[1] #grid point member fraction
            
        out = None
        
        with cmodel:
           # update member reactions bounds if required:
           if update_bounds: 
                print('updating reaction bounds ...')
                # In the case that all objectives are zero, all member fractions are nan
                # Here we set fractions to zero to avoid errors setting bounds
                if np.all(np.isnan(pfrac)): 
                    pfrac = np.array([0.0]*pfrac.size)                
                    
                # reactions are assign to each community member
                self.get_member_reactions()
                for i, member in enumerate(self.prefixes):
                    mfrac = pfrac[i]
                    mrxns = self.member_rxns[member]
           
                    # rxn bounds are updated, accounting for members fractions in the community 
                    
                    for rid in mrxns:
                        r = cmodel.reactions.get_by_id(rid)
                        old_bounds = r.bounds
                        r.bounds =(old_bounds[0] * mfrac, old_bounds[1] * mfrac)

           # fix member objectives to grid point value:
           for ix, member_objectives in enumerate(self.objectives):    
                if len(member_objectives) != 1:
                    raise RuntimeError("Warning: More than one reaction in %s objective function. Not supported!!" 
                      % self.prefixes[ix])        
                #new bounds for member ix objective function reaction:
                new_bounds = (point[ix],point[ix])    
                #newGrid NJ
                #print(point[ix])

            
                #change bounds for each objective reaction
                for rid in member_objectives.keys(): # member_objectives should be single key dictionary
                    rxn = cmodel.reactions.get_by_id(rid)
                    rxn.bounds = new_bounds

                    #set one of objective reactions as community objective
                    #cmodel.objective = rid #commented since the model comes with an objective
            
           
           if analysis == 'feasibility': 
                #check point feasibility
                error_value = -1000
                ob = cmodel.slim_optimize(error_value = error_value)  
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
                
                rxn_fva = flux_variability_analysis(cmodel,reaction_list= self.rxn2cluster)
                
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
        if self.pfractions is None:
            self.get_points_distribution()
        
        point_array     = self.points               
        pfraction_array =self.pfractions
        
        #Option 'feasibility':
        if analysis == 'feasibility':
                
            #step 2: run feasibility analysis for all grid points
            npoints = point_array.shape[0]       
            self.feasible_points = self.check_feasible(point_array, pfraction_array, 
                                                       update_bounds=update_bounds, **kwargs)
            
            npoints = point_array.shape[0]
            nfeasible = sum(self.feasible_points)
            print("grid feasible points: %d / %d" % (nfeasible, npoints))
            
        elif analysis == 'qual_fva':
                
            #step 2: check for previously calculated feasible points  
            feasible_points = self.feasible_points
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
                
    def change_reaction_bounds(self, rid, new_bounds):
        cmodel = self.community_model
        rxn = cmodel.reactions.get_by_id(rid)
        old_bounds = rxn.bounds
        rxn.bounds = new_bounds
        return old_bounds


    # distinto                    
    def build_grid(self,expand = True, numPoints=10, drop_zero = True):
        
        #polytope vertex
        #polytope_vertex = self.get_polytope_vertex(expand=expand)
        
        #polytope from vertex
        #hull = Delaunay(polytope_vertex)
        
        # "rectangular" grid
        #maxs = np.max(polytope_vertex, axis=0)
        #mins = np.min(polytope_vertex, axis=0) #should be vector of zeros
        #max_com = self.cmodel.slim_optimize() #assuming objective function of the model is community growth
        
        #compute max_com by relaxing constraints such as ATPM
        with self.community_model:
            for rxn in self.community_model.reactions:
                if rxn.lower_bound > 0:
                    rxn.lower_bound = 0
            
            max_com = self.community_model.slim_optimize()
        maxs = [1 ,max_com]
        print('Maximum community:'+str(max_com))
        mins = [0,0] #hardcoded 2D 
        size = self.size
        
        #Modify this to have a matrix of nxn points rather than a step (using com_growth and fraction as axis)
        #alternative: define a different step based on points to have
        slices = [np.linspace(mins[i], maxs[i], numPoints) for i in range(size)]
        #slices = [slice(mins[i],maxs[i],step) for i in range(size)]
        #rgrid = np.mgrid[slices]

        print(slices[0])
        print(slices[1])
        rgrid = np.array(np.meshgrid(slices[0], slices[1]))#.T.reshape() #NJ
        print(rgrid)
        #rgrid2columns = [rgrid[i,:].ravel() for i in range(size)]
        rgrid2columns = [rgrid[i,:].ravel() for i in range(size)]
        # array of grid points (x,y,z,...)
        positions = np.column_stack(rgrid2columns)
        #polytope intersection    
        #inside = hull.find_simplex(positions)>=0
        
        # storing grid points inside polytope
        #points = positions[inside] 
        points = positions
        limits = (mins,maxs)
        if drop_zero:
            points = points[1:]
            mins   = np.min(points, axis=0)
            maxs   = np.max(points, axis=0)
        
        
        
        self.points  = points
        #self.step    = step
        self.limits  = (mins,maxs)

    def get_selected_points_distribution(self, prange = None):
        
        if self.points is None:
            raise RuntimeError('Grid points are not set yet!')
            
        
        points = self.points
        #NJ new grid
        #points[0]: f_i
        #points[1]: com_u
        
        if prange is not None:
            points = points[prange]
        
        #com_mu = np.sum(points,axis =1)
        #pfractions = np.apply_along_axis(lambda a, b : a/b, 0, points, com_mu)     
        pfractions = np.array([[p[0], 1-p[0]] for p in points]) #assuming two organisms
        return pfractions        
        
    def get_points_distribution(self):
        pfractions = self.get_selected_points_distribution(prange = None)
        self.pfractions = pfractions

    def calculate_community_growth(self, feasible=False): #NJ DELETE THIS FUNCTION
        cgrowth = np.sum(self.points,axis=1) 
        if feasible:
            if self.feasible_points is None:
                print('feasible points have not been previously established! Returning values for all points')
            else:
                cgrowth =cgrowth[self.feasible_points]
                       
        return cgrowth 
     
    def get_member_reactions(self):
        member_rxns = {x:[] for x in self.prefixes}
        for r in self.community_model.reactions:
            for member in self.prefixes:
                if r.id.startswith(member):
                    member_rxns[member].append(r.id)
                    break
        self.member_rxns = member_rxns                  
    
    
    def get_blocked_reactions(self):
        blocked = cobra.flux_analysis.find_blocked_reactions(self.community_model)
        return blocked     
        
    def get_non_blocked_reactions(self):
        blocked = cobra.flux_analysis.find_blocked_reactions(self.community_model)
        all_ids = [x.id for x in self.community_model.reactions]
        non_blocked = set(all_ids).difference(set(blocked))
        self.non_blocked = non_blocked
        
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
        dvector = distance.pdist(z,distance_metric)
        dmatrix = distance.squareform(dvector)
        
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
        
    def write_fca_input(self,prefix,file_dir, discard_zero_bound=True):
     
        file_name = "%s/%s_fca_input.mat" % (file_dir,prefix)
        cmodel = self.community_model 
            
        rxns = cmodel.reactions
        mets = cmodel.metabolites
        stoich_mat = create_stoichiometric_matrix(cmodel)
        rids = np.array(rxns.list_attr("id"))
        mids = np.array(mets.list_attr("id"))
        rev  = np.array(rxns.list_attr("reversibility"))*1
    
        #discard reactions from pool and other members and also reactions from prefix with zero lower and upper bounds 
        #these last reactions are added to blocked.
        
        to_discard = []
        blocked = [] 
                
        for ix in range(len(rids)):
            rid = rids[ix]
            if not rid.startswith(prefix):
                to_discard.append(ix)
            else:        
                if discard_zero_bound:            
                    r = cmodel.reactions.get_by_id(rid)
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
            if not mid.startswith(prefix):
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
        
        mat2 = OrderedDict()
        #mat2['bound_blocked'] = _cell(blocked)
        varname1 = 'fca_input'
        varname2 = 'bound_blocked'
        #varname2 = "%s_bound_blocked" % prefix
        #sio.savemat(file_name, {varname1: mat, varname2:mat2}, oned_as="column")
        sio.savemat(file_name, {varname1: mat, varname2:_cell(blocked)}, oned_as="column")
        print("%s FCA's input in %s" % (prefix, file_name))
        print("   stoichiometric matrix : %s" % str(stoich_mat.shape))
        
    def store_fca_results(self,prefix,fca_file):

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
                
        self.coupled_rxns[prefix] = coupled_sets
        self.member_blocked[prefix]= blocked_ids
        total_rxns = len(rxn_ids)+len(bound_blocked)
        print("Flux coupling results for member %s stored:" % prefix)
        print("   Total reactions: %d" % total_rxns)
        print("   Fully coupled reaction sets: %d" % len(coupled_sets))
        print("   Blocked reactions: %d" % len(blocked_ids))      
        print("-")
        
    def set_cluster_reactions(self):
        
        # if FCA has been performed and results stored for all members, reactions for fva and clustering are reduced accordingly.
        # Otherwise, non-blocked reactions are obtained and ALL those reactions are used.       

        
        coupled_dicts =  list(self.coupled_rxns.values())
              
        if coupled_dicts.count(None) != 0: # at least one member without FCA results       
            print("Missing FCA results")
            print("Using non-blocked reactions only")
            self.get_non_blocked_reactions()  
            rxn2cluster = list(self.non_blocked)             

        else:      
            accounted = []      
            coupled_rep = []
              
            # blocked reactions are not considered for fva and clustering   
            for prefix in self.member_blocked:
                blocked = self.member_blocked[prefix]
                accounted = accounted + blocked
              
            # Only one reaction from each fully coupled set is used for fva and clustering      
            for prefix in self.coupled_rxns:
                coupled_sets_dict = self.coupled_rxns[prefix]
                # rxn representative for coupled set:   
                coupled_rep  = coupled_rep + list(coupled_sets_dict.keys())
                coupled_sets = list(coupled_sets_dict.values())
                accounted = accounted + [rxn for cset in coupled_sets for rxn in cset]
        
            # Reactions for fva and clustering are those representing coupled sets and those not in any member (pool reactions)                   
            all_rids = [r.id for r in self.community_model.reactions]
            missing = set(all_rids).difference(set(accounted))
        
            rxn2cluster = list(missing) + coupled_rep       
        
        rxn2cluster.sort()
        self.rxn2cluster =  rxn2cluster
        print("Total reactions considered for fva and clustering: %d" % len(self.rxn2cluster))

        