"""
A files with a bunch of useful templates, to make
testing/doing stuff more easy.
"""

import bioreaction
from bioreaction.model.data_containers import *



def GeneralMedSim():
    """
    A medium sim to show it's utility, and do tests
    on. 😀 
    No control... yet
    """
    prot_species = [Species(f"Protein {i}") for i in range(3)]
    rna_species = [Species(f"RNA {i}") for i in range(3)]

    all_species = prot_species + rna_species


    rna_reactions = []
    def get_rna_r(i, fr, rr):
        return Reaction(input = [], output = [rna_species[i]], forward_rate=fr, reverse_rate= rr)
    
    rna_reactions.append(get_rna_r(0, 2, 0.5))
    rna_reactions.append(get_rna_r(1, 5, 3))
    rna_reactions.append(get_rna_r(2, 0.5, 0.1))

    def get_prot_rs(i, make_rate, decay_rate):
        make_react = Reaction(input = [rna_species[i]],
            output=[rna_species[i],prot_species[i] ], forward_rate= make_rate)
        decay_react = Reaction(input = [prot_species[i]], output=[], forward_rate= decay_rate)
        return [make_react, decay_react]

    prot_reactions = []
    [prot_reactions.append(x) for x in get_prot_rs(0, 10, 0.5)]
    [prot_reactions.append(x) for x in get_prot_rs(1, 4, 0.1)]
    [prot_reactions.append(x) for x in get_prot_rs(2, 1, 0.1)]


    all_reactions = rna_reactions + prot_reactions

    other_factors = [OtherFactor("Transcription Factor"), 
                     OtherFactor("Translation Factor"),
                     OtherFactor("RNA Decay Factor")]

    ou_effects = [OUProcess.scale_std_init(other_factors[0], 2.0, 0.15),
                  OUProcess.scale_std_init(other_factors[1], 0.5, 0.2),
                  OUProcess.scale_std_init(other_factors[2], 1.5, 0.1)]
    
    # There should be a ncie way of doing this..
    extra_factors = [ExtraReactionEffect(factor = other_factors[0], target_reaction=rna_reactions[0]
                                        , forward_strength=1.0, backward_stength= 0.0),
                    ExtraReactionEffect(factor = other_factors[0], target_reaction=rna_reactions[1]
                                        , forward_strength=1.4, backward_stength= 0.0),
                    ExtraReactionEffect(factor = other_factors[0], target_reaction=rna_reactions[2]
                                        , forward_strength=0.7, backward_stength= 0.0),
                    ExtraReactionEffect(factor = other_factors[1], target_reaction=prot_reactions[0]
                                        , forward_strength=1.2, backward_stength= 0.0),
                    ExtraReactionEffect(factor = other_factors[1], target_reaction=prot_reactions[2]
                                        , forward_strength=1.4, backward_stength= 0.0),
                    ExtraReactionEffect(factor = other_factors[1], target_reaction=prot_reactions[4]
                                        , forward_strength=0.5, backward_stength= 0.0),
                    ExtraReactionEffect(factor = other_factors[2], target_reaction=rna_reactions[0]
                                        , forward_strength=0.0, backward_stength= 1.0),
                    ExtraReactionEffect(factor = other_factors[2], target_reaction=rna_reactions[1]
                                        , forward_strength=0.0, backward_stength= 0.5),
                    ExtraReactionEffect(factor = other_factors[2], target_reaction=rna_reactions[2]
                                        , forward_strength=0.0, backward_stength= 1.2)
                    ]

    impulses = [Impulse(target= rna_species[0], delta_target=5.0, time = 2.4, impulse_width=0.0),
                Impulse(target= prot_species[1], delta_target=65, time = 1.0, impulse_width=0.2)
                ]

    modelly = MedModel( species=all_species, reactions = all_reactions, other_factors=other_factors,
                        reaction_extrinsics=extra_factors, impuluses=impulses, controllers=[],
                        ou_effects=ou_effects)

    return modelly






