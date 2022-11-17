
import unittest
from bioreaction.model.data_containers import QuantifiedReactions, Species
from bioreaction.model.data_tools import construct_model


class TestModel(unittest.TestCase):

    def test_construct_model(self):
        fake_config = {
            "reactions": {
                'inputs': [['A', 'B']],
                'outputs': None
            }}
        model = construct_model(fake_config)
        reaction_input = model.reactions[0].input
        reaction_output = model.reactions[0].output
        self.assertEqual(type(reaction_input), list)
        self.assertEqual(len(reaction_input), 2)
        self.assertEqual(type(reaction_input[0]), Species)
        self.assertEqual(type(reaction_output[0]), Species)

        # TODO: write a test that makes sure that each component within the output Species is also a Species
        # and that that Species is the same object as the one in the inputs list

    def test_species_concentrations(self):
        fake_config = {
            "reactions": {
                'inputs': [['A'], ['A', 'B']],
                'outputs': [['A_degraded'], ['AB']]
            }, 
            "starting_concentration": {
                "A": 10,
                "B": 1,
                "A_degraded": 0,
                "AB": 1
            },
            "forward_rates": [
                1.5, 1.4
            ],
            "reverse_rates": [
                0, 0.5
            ]
        }

        model = construct_model(fake_config)
        qreactions = QuantifiedReactions()
        qreactions.init_properties(model, fake_config)

        self.assertEqual(qreactions.quantities[0], 
            fake_config['starting_concentration']['A'])


if __name__ == '__main__':
    unittest.main()
