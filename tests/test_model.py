
import unittest
from bioreaction.model.data_containers import Species
from scripts.playground.simple_reaction import construct_model


class TestModel(unittest.TestCase):

    def test_construct_model(self):
        fake_config = {
            "reactions": {
                'inputs': [['A', 'B']],
                'outputs': None
            }}
        model = construct_model(fake_config)
        reaction_input = model.reactions[0].input
        reaction_input = model.reactions[0].output
        self.assertEquals(type(reaction_input[0]), Species)
        self.assertEquals(type(reaction_input[1]), Species)

        # TODO: write a test that makes sure that each component within the output Species is also a Species
        # and that that Species is the same object as the one in the inputs list


if __name__ == '__main__':
    unittest.main()
