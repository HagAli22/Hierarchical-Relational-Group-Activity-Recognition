# Models package
# ==============

# Person classifier (Stage 1)
from .person_classifer import Person_Classifer

# Alias for backward compatibility
person_classifer = Person_Classifer

# Non-temporal models (Stage 2 baselines)
from .non_temporal_model.B1_NoRelations import B1_NoRelations, collate_group_fn

# Aliases
B1NoRelations = B1_NoRelations
