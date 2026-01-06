from typing import cast
from cobra import Model, Reaction


def clone_with_modified_bounds(model: Model, bounds_dict: dict[str, tuple[int, int]]) -> Model:
    '''Returns a copy of the given COBRA model with reaction bounds updated. The reactions specified in "bounds_dict" are replaced by the provided (lower, upper) tuples.
    The original model is not modified.'''
    new_model = model.copy()

    for reaction_id, bounds in bounds_dict.items():
        reaction = cast(Reaction, new_model.reactions.get_by_id(reaction_id))
        reaction.bounds = bounds

    return new_model


