import numpy as np
from functools import reduce
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ecosystem_base import BaseEcosystem


class EcosystemGrid():
    def __init__(self, base_ecosystem: "BaseEcosystem", points, feasible_points, pfractions, grid_step):
        self.ecosystem = base_ecosystem

        self.points = points
        self.feasible_points = feasible_points
        self.pfractions = pfractions

        self.step = grid_step
        self.limits: tuple = tuple()
        
    @property
    def member_model_ids(self):
        return self.ecosystem.member_model_ids
    
    @property
    def size(self):
        return self.ecosystem.size


    # GRID CONSTRUCTION =======================================================================


    # distinto, se agrega ignore_maint, los comentarios son distintos            
    def build_grid(self, numPoints: int = 10, drop_zero: bool = True, ignore_maint: bool = True) -> None:
                
        #compute max_com by relaxing constraints such as ATPM
        with self.ecosystem.community_model as community_model:
            if ignore_maint:
                for rxn in community_model.reactions:
                    if rxn.lower_bound > 0:
                        rxn.lower_bound = 0
            
            max_com: float = community_model.slim_optimize()

        maxs = [1, max_com]
        print(f'Maximum community: {max_com}')
        mins = [0, 0] #hardcoded 2D 
        size = self.size
        
        #Modify this to have a matrix of nxn points rather than a step (using com_growth and fraction as axis)
        #alternative: define a different step based on points to have
        slices = [np.linspace(mins[i], maxs[i], numPoints) for i in range(size)]
        #slices = [slice(mins[i],maxs[i],step) for i in range(size)]
        #rgrid = np.mgrid[slices]

        print(f"slice 0: {slices[0]}")
        print(f"slice 1: {slices[1]}")

        rgrid = np.array(np.meshgrid(slices[0], slices[1]))#.T.reshape() #NJ
        print(rgrid, type(rgrid))
        #rgrid2columns = [rgrid[i,:].ravel() for i in range(size)]
        rgrid2columns = [rgrid[i,:].ravel() for i in range(size)]
        # array of grid points (x,y,z,...)
        points = np.column_stack(rgrid2columns)
        print(f"points type: {type(points)}")
     
        if drop_zero:
            points = points[1:]
            mins   = np.min(points, axis=0)
            maxs   = np.max(points, axis=0)
        
        self.points  = points
        self.limits  = (mins, maxs)


    def set_points_distribution(self, prange = None):
        
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
        self.pfractions = np.array([[p[0], 1-p[0]] for p in points]) #assuming two organisms
            

    def get_2D_slice(self, model_ids: list, fixed_values: list): #NJ DELETE THIS FUNCTION
        if self.size - len(model_ids) != 2:
            raise RuntimeError("Two members with non-fixed values required! No more, no less.")
        
        
        members_to_fix = range(len(model_ids))
        #valores mas cercanos en la grilla a fixed_values
        closest_values = [self._get_closest_grid_value(model_ids[i], fixed_values[i]) for i in members_to_fix]
        fixed_member_indexes = [self.member_model_ids.index(x) for x in model_ids]
        
        
        grid_points = self.points
        #grid_points_values = self.points_values # aqui va matriz de puntos x reacciones con valores cualitativos
                                         # calculados a partir de fva. 
        
        #indices de puntos en grilla donde el valor una dimension es igual a su closest_values
        filtered_indexes_list =  [np.where(grid_points[:,fixed_member_indexes[i]] == closest_values[i])[0] for i in members_to_fix]
        
        #interseccion de indices, i.e., indices slice
        slice_indexes =  reduce(np.intersect1d, filtered_indexes_list)
        #slice 2D de la grilla
        #filtered_points = grid_points[slice_indexes,:] 
        #filtered_values = grid_points_values[slice_indexes,:]
        
        free_members = list(set(self.member_model_ids).difference(set(model_ids))) # model ids of members with non-fixed objective values
        free_member_indexes = [self.member_model_ids.index(x) for x in free_members]
        #Puntos se reducen a las dimensiones de los free members:
        #slice_points = filtered_points[:,free_member_indexes]
        #slice_points_values = filtered_values[:,free_member_indexes]
          
        #return slice_points, slice_points_values , free_member_indexes
        return [slice_indexes, free_member_indexes]    
    

    # used by ecosystem_plot
    def _get_closest_grid_value(self, model_id, fixed_value):
        member_index = self.member_model_ids.index(model_id)
        member_min = self.limits[0][member_index]
        member_max = self.limits[1][member_index]
        
        if fixed_value < member_min or fixed_value > member_max:
            raise RuntimeError("Value %d for %s out of range" % (fixed_value, model_id))
        
        shifted_value = fixed_value - member_min
        n_steps = shifted_value//self.step          
        
        p1 = n_steps * self.step
        p2 = (n_steps + 1) * self.step  
        
        if (shifted_value - p1) < (p2 - shifted_value):
            closest_value = member_min + p1
        else:
            closest_value = member_min + p2        
        
        return closest_value    


    def resolve_2D_slice(self, model_ids, fixed_values):
        # get full 2D slice:   
        if self.size == 2:
            if len(model_ids) > 0:
                print("Only two members in community!!") 
                print("Full grid will be plotted and fixed values for %s will be ignored..." % str(model_ids))
            free_member_model_ids = self.member_model_ids
            free_member_indexes = [0,1]
            full_slice_indexes = np.arange(len(self.points)) 
            
        else:              
            full_slice_indexes, free_member_indexes = self.get_2D_slice(model_ids, fixed_values)
            free_member_model_ids = [self.member_model_ids[x] for x in free_member_indexes]

        return full_slice_indexes, free_member_indexes, free_member_model_ids


    def get_slice_points(self, full_slice_indexes, free_member_indexes):

        if self.feasible_points is not None: 
            feasible_indexes    = np.where(self.feasible_points)[0]
            feasible_points     = self.points[feasible_indexes]
            feasible_pfractions = self.pfractions[feasible_indexes]
                
            slice_indexes    = np.isin(feasible_indexes, full_slice_indexes)
            slice_points     = feasible_points[slice_indexes,:][:,free_member_indexes]  
            slice_pfractions = feasible_pfractions

        else:
            slice_indexes    = full_slice_indexes
            slice_points     = self.points[full_slice_indexes,:][:,free_member_indexes] 
            slice_pfractions = self.pfractions[full_slice_indexes]
            
        return slice_points, slice_pfractions, slice_indexes
    

    def _calculate_community_growth(self, feasible=False): #NJ DELETE THIS FUNCTION
        cgrowth = np.sum(self.points,axis=1) 
        if feasible:
            if self.feasible_points is None:
                print('feasible points have not been previously established! Returning values for all points')
            else:
                cgrowth = cgrowth[self.feasible_points]
                       
        return cgrowth     


    def get_polytope_vertex(self, expand: bool = True):
  
        """
        polytope: pareto front + axes segments + extra segments perpendicular to axes dimensions where 
        pareto solutions don't reach 0 values. 
        (assumption: objective functions can only take positive values)
        """
        pareto_front = self.ecosystem.community._get_pareto_front()

        #2. origin
        ov = np.zeros((1, self.size))
        
        if expand == True:
            #3. Extra points that close polytope (only if points (0,0,0,...,xi_max,0,...0) are not pareto front 
            # points but they are feasible) 
            
            #MP: si hay que incluir estos puntos significa que hay miembros que son givers: i.e. pueden crecer
            # a su máxima tasa y aguantar que otros miembros crezcan también
            # si un punto (0,0,0,...,xi_max,0,...0) no es factible entonces el miembro i necesita que otros crezcan 
            # para poder crecer (needy).
            
            #3.1 Check if points  (0,0,0,...,xi_max,0,...0) are part of the pareto front
            n = self.size - 1
            all_zeros_but_one = np.argwhere(np.sum(pareto_front == 0,axis=1)==n) # index of (0,0,...,xi_max,0,0...0) points
            all_zeros_but_one = all_zeros_but_one.flatten()
    
            # indexes i of non-zero member in (0,0,...,xi_max,0,0...0) pareto points, 
            # i.e. members that are not givers nor needy. 
            non_zero_dims =  np.argmax(pareto_front[all_zeros_but_one,:], axis = 1) 
            
            # givers and/or needy members:
            givers_or_needy_indexes = np.setdiff1d(np.array(range(self.size)), non_zero_dims) 
            gn_total= len(givers_or_needy_indexes)    
        
            #3.2 Check if non-pareto points (0,0,0,...,xi_max,0,...0) are feasible 
            if gn_total >0:
                # max values for giver_or_needy members:
                max_vals = np.max(pareto_front, axis=0)
                cpoints = np.diag(max_vals)
                to_check = cpoints[givers_or_needy_indexes,:]
                
                are_feasible = self.ecosystem.analyze.check_feasible(to_check)
                
                ev = to_check[are_feasible, :] 

                polytope_vertex = np.concatenate((pareto_front,ov,ev), axis=0)
            else: 
                polytope_vertex = np.concatenate((pareto_front,ov), axis=0)
        
        else:
                polytope_vertex = np.concatenate((pareto_front,ov), axis=0)

        return polytope_vertex 
    

    
