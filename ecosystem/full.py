from cobra import Metabolite, Reaction, Model
from cobra.util.solver import linear_reaction_coefficients

from ecosystem.base import BaseEcosystem
from typing import Any, cast

Numerical = int |float


class FullEcosystem(BaseEcosystem):
    def __init__(self, member_models: list[Model], member_model_ids: list[str], 
                 community_name: str = "community", community_id: str = "community",
                 pool_bounds: tuple[Numerical, Numerical] = (-1000,1000), keep_members: bool = True, print_report: bool = True, solver: str = 'gurobi'):
        super().__init__(community_name, community_id)

        self.size = len(member_models)
        self.member_model_ids = member_model_ids
        self.conflicts: list[str] = [] # unique to this class, stores the ids of conflicting metabolites
      
        print("\n0. Copying member models ...")
        self.member_models: list[Model] = [model.copy() for model in member_models]
    
        print("\n1. Storing exchanged metabolites information ...")
        self._set_members_exchange_info()
        
        print("\n2. Updating member model objects ids ...")
        self._update_all_members()
        
        print("\n3. Storing member objectives ...")  
        self._set_members_objectives()

        print("\n4. Merging member models ...")
        self._merge_all_models(solver)
              
        print("\n5. Creating pool compartment, metabolites and reactions ...")
        self._add_pool(pool_bounds)
        
        print("\n6. Updating compartment names ...")
        self._update_compartment_names()

        print("\nDone. Community model created")

        # load all other modules
        #self._build_modules() 

        # --- optional steps ----
        self._print_build_report(pool_bounds, print_report) # Print summary report if print_report is True
        self._clean_members(keep_members)                   # Member models are kept unless keep_members is False 
    

    # ======================================================================== METHODS ========================================================================
    

    # --------------------------------------------------- step 1 ---------------------------------------------------

 
    def _check_conflicts(self, exchange_metabolite_info: dict[str, dict[str, Any]]) -> None:
        '''Checks for inconsistencies in exchanged metabolites among member models and fill missing values in-place'''
        
        conflicts = list()

        # saves all exchange metabolite information on a dictionary of form 
        for metabolite_id in exchange_metabolite_info:
            formulas = set()
            charges  = set()
            names    = set()

            metabolite_info = exchange_metabolite_info[metabolite_id]

            for model_id in self.member_model_ids:
                # formulas, charges and names get registered in case the metabolite is present in model labeled by model_id
                if model_id in metabolite_info:
                    formulas.add(metabolite_info[model_id]['formula'])
                    charges.add(metabolite_info[model_id]['charge'])
                    names.add(metabolite_info[model_id]['name'])
                else:
                    # otherwise, the information gets filled in
                    exchange_metabolite_info[metabolite_id][model_id] = {
                        'm_id'  : None,
                        'name'  : None,
                        'formula': None,
                        'charge' : None,
                        'ex_id' : None,
                        'bounds' : None,                                       
                    } 
    
            # checks using sets if a metabolite has two or more distinct formulas, charges and names
            if len(formulas) > 1 or len(charges) > 1 or len(names) > 1:
                conflicts.append(metabolite_id) 
                print("Conflicts for exchange metabolite %s:" % metabolite_id)
                print(f"Formulas: {formulas}")
                print(f"Charges: {charges}")
                print(f"Names: {names}")
                print("----------")       
            
        self.conflicts = conflicts
        self.community.exchange_metabolite_info = exchange_metabolite_info


    def _set_members_exchange_info(self, retrieve_all_metabolites: bool = False) -> None:
        '''
        Information about exchanged metabolites in each member model information is stored in 'exchange_metabolite_info'.
        Exchange metabolites with conflicting data between member models (different formulas, charges, or names) are stored in 'conflicts'
        '''
        if self.community.exchange_metabolite_info:
            print('Exchange information already stored.')     
            return
        
        # each metabolite has an entry on this dictionary, which stores another dictionary where its information lies
        exchange_metabolite_info: dict[str, dict[str, Any]] = dict()

        # stores all exchange metabolite information on a dictionary
        for index, model in enumerate(self.member_models):
            model_id = self.member_model_ids[index]

            for exchange_reaction in model.boundary: 
                # only one metabolite per reaction restriction
                if not retrieve_all_metabolites and len(exchange_reaction.metabolites) > 1:
                    print("Not supported! Use standard of single metabolite exchanges!")
                    raise Exception("More than one metabolite in exchange reaction %s from %s model" % (exchange_reaction.id, model_id))
                            
                for metabolite in exchange_reaction.metabolites:
                    if metabolite.id not in exchange_metabolite_info:
                        exchange_metabolite_info[metabolite.id] = dict()

                    exchange_metabolite_info[metabolite.id][model_id] = {
                        'm_id':     "%s_%s" % (model_id, metabolite.id),
                        'name':     metabolite.name,
                        'formula':  metabolite.formula,
                        'charge':   metabolite.charge,
                        'ex_id':    "%s_%s" % (model_id, exchange_reaction.id),
                        'bounds':   exchange_reaction.bounds,                      
                    } 

        self._check_conflicts(exchange_metabolite_info)


    # --------------------------------------------------- step 2 ---------------------------------------------------


    def _update_all_members(self) -> None:
        """ Updates all member's models by renaming their metabolite and reaction ids, and also 
            their compartment names and ids. 
            The corresponding community member prefixes are used:

                original_rxn_id           -> (member_model_id)_(original_rxn_id)

                original_met_id           -> (member_model_id)_(original_met_id)

                original_compartment_id   -> (member_model_id)_(original_compartment_id)

                original_compartment_name -> (member_model_id) (original_compartment_name)
         """
        for member_index in range(self.size):
            model = self.member_models[member_index]
            model_id = self.member_model_ids[member_index]
            comp_dict = dict()        

            #Updating reaction ids
            for reaction in model.reactions:
                reaction_id = "%s_%s" % (model_id, reaction.id) #NJ changed %s:%s to %s_%s
                reaction.id = reaction_id

            # Updating metabolite ids and compartment ids and names        
            for metabolite in model.metabolites:
                metabolite_id = "%s_%s" % (model_id, metabolite.id) 
                metabolite.id = metabolite_id

                metabolite_compartment = "%s_%s" % (model_id, metabolite.compartment) 
                if metabolite_compartment not in comp_dict:
                    compartment_name = model.compartments[metabolite.compartment] 
                    if len(compartment_name) == 0: # in case compartment doesn't have a name
                        compartment_name = metabolite.compartment

                    comp_dict[metabolite_compartment]= "%s %s" % (model_id, compartment_name)

                metabolite.compartment = metabolite_compartment  

            model.compartments = comp_dict
            model.repair()


    # --------------------------------------------------- step 3 ---------------------------------------------------
    

    def _set_members_objectives(self) -> None:
        '''Store member models original objectives {rxn: coef}, before merging models'''
        for model_index, model in enumerate(self.member_models):
            member_lr_coefs = linear_reaction_coefficients(model)
            if len(member_lr_coefs) != 1:
                raise RuntimeError("More than one reaction in %s objective function. Not supported! Set single objective reaction" % self.member_model_ids[model_index]) 
                            
            member_objectives = {r.id: coeff for r, coeff in member_lr_coefs.items()} # should be single key,value dict   
            self.objectives.append(member_objectives)


    # --------------------------------------------------- step 4 ---------------------------------------------------


    def _merge_all_models(self, solver: str) -> None:
        '''Create new cobrapy Model for the community by merging member models. Use one of the members' objective function as default objective for the community model'''
        community_model = self.community_model
        for model in self.member_models:
            community_model.merge(model)
            
        all_objectives = dict()
        for objective_dict in self.objectives:
    
            for reaction_id in objective_dict.keys():
                reaction = community_model.reactions.get_by_id(reaction_id)
                all_objectives[reaction] = objective_dict[reaction_id] 
            
        community_model.solver = solver
        community_model.objective = all_objectives


    # --------------------------------------------------- step 5 ---------------------------------------------------


    def _add_pool_metabolites(self) -> list[Metabolite]:
        '''Creating and adding pool metabolites to community model.'''

        exchange_metabolite_info = self.community.exchange_metabolite_info
        print(exchange_metabolite_info)
        pool_metabolites = list()

        for metabolite_id in exchange_metabolite_info:
            pool_id = "%s_pool" % metabolite_id    

            for model_id in self.member_model_ids: # uses first available model 
                info = exchange_metabolite_info[metabolite_id][model_id]

                if info['m_id'] is not None: # info is taking from one of the member models
                    name     = info['name']
                    charge   = info['charge']
                    formula  = info['formula']
                    
                    if metabolite_id in self.conflicts:
                        print("Warning: Adding pool metabolite %s with conflicting information in member models" % metabolite_id)
                        print("Using data from %s model" % model_id)  

                    pool_metabolite = Metabolite(pool_id, formula = formula, charge = charge, name = name, compartment = 'pool')
                    pool_metabolites.append(pool_metabolite)                      
                    break   
        
        self.community_model.add_metabolites(pool_metabolites)    
        return pool_metabolites    


    def _add_pool_reactions(self, pool_metabolites: list[Metabolite], pool_bounds: tuple[Numerical, Numerical]) -> None:
        '''Creating and adding pool exchange reactions to community model.'''   

        pool_reactions = list() 
        for pool_metabolite in pool_metabolites:
            metabolite_id = pool_metabolite.id.replace('_pool', '')
            exchange_reaction = Reaction(
                id   = "EX_%s" % metabolite_id,
                name = "Pool %s exchange" % pool_metabolite.name,
                subsystem = "Exchange",
                lower_bound = pool_bounds[0],
                upper_bound = pool_bounds[1]
            )
            exchange_reaction.add_metabolites({pool_metabolite: -1.0})        
            pool_reactions.append(exchange_reaction)
            
        self.community_model.add_reactions(pool_reactions)


    def _redirect_exchanges_to_pool(self) -> None:
        '''Updates original exchange reactions from member models: from met_e:model1 <--> to met_e:model1 <--> met_e:pool.
        Setting bounds to pool bounds.'''

        community_model = self.community_model
        exchange_metabolite_info = self.community.exchange_metabolite_info

        for metabolite_id, metabolite_info in exchange_metabolite_info.items():
            pool_id = "%s_pool" % metabolite_id 
            pool_metabolite = cast(Metabolite, community_model.metabolites.get_by_id(pool_id))
            
            for info in metabolite_info.values():
                if info['m_id'] is not None:
                    metabolite = cast(Metabolite, self.community_model.metabolites.get_by_id(info['m_id']))
                    exchange_reaction = cast(Reaction, community_model.reactions.get_by_id(info['ex_id']))
                    coeff = exchange_reaction.metabolites[metabolite]
                    exchange_reaction.add_metabolites({pool_metabolite: -coeff})

   
    # By default pool exchanges are set with bounds (-1000,1000) and individual members' exchange
    # reaction bounds are set to the same values. 
    def _add_pool(self, pool_bounds: tuple[Numerical, Numerical] = (-1000,1000)) -> None:
        """ 
        Adds pool to community model (including compartment, shared pool mets, pool exchanges
                                      and update of member exchange reactions)
         By default pool exchanges are set with bounds (-1000,1000) 
         Individual member exchange reactions bounds are not modified!      
        """
        community_model = self.community_model
        
        compartments = community_model.compartments
        compartments['pool'] = "Community pool"
        
        pool_metabolites = self._add_pool_metabolites()

        self._add_pool_reactions(pool_metabolites, pool_bounds)

        community_model.compartments = compartments
        community_model.repair()

        self._redirect_exchanges_to_pool()
        
      
    # --------------------------------------------------- step 6 --------------------------------------------------- 


    def _update_compartment_names(self) -> None:
        '''Add compartment names. They get lost during merge (step 4)'''
        community_compartments = self.community_model.compartments
        for model in self.member_models:
            for compartment in model.compartments:
                community_compartments[compartment] = model.compartments[compartment]
        self.community_model.compartments = community_compartments    


    # --------------------------------------------------- optional ---------------------------------------------------


    def _print_build_report(self, pool_bounds: tuple[Numerical, Numerical], print_report: bool = True) -> None:
        """ Prints community construction report """
        if not print_report:
            return
        
        community_model = self.community_model
        models = self.member_models
        model_ids = self.member_model_ids
        print("Created community model from %d member models." % self.size)               
 
        print("General stats:")
        if models is not None:
            for i in range(self.size):
                print("model (%d):" % i) 
                print( "\t id = %s, name = %s , model_id = %s" % (models[i].id,models[i].name, model_ids[i]))

                compartments = len(models[i].compartments)
                reactions = len(models[i].reactions)
                exchange_metabolites = len(models[i].boundary)
                print( "\t\t compartments = %d" % compartments)
                print( "\t\t reactions = %d" % reactions)
                print( "\t\t exchange metabolites = %d" % exchange_metabolites)
                
        print("community model:")
        print( "\t id = %s, name = %s " % (community_model.id, community_model.name))
        print( "\t\t reactions = %d" % len(community_model.reactions))
        print( "\t\t exchange metabolites = %d" % len(self.community.exchange_metabolite_info)) 
        print( "\t\t compartments = %d" % len(community_model.compartments))        
        print("Exchange metabolite conflicts (formula, charge, name, id) = %d" % len(self.conflicts))   
        print("Community exchange reaction bounds: %s" % str(pool_bounds))
    

    def _clean_members(self, keep_members: bool = False) -> None:
        """ Deletes member individual models"""
        if keep_members:
            return

        self.member_models.clear()
        
