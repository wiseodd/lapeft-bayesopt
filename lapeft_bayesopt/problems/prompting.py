
class PromptBuilder:
    """
    Base class for building prompts. To be derived (per dataset/problem if needed).

    Parameters:
    -----------
    kind: str
        The name of the prompt type
    """
    def __init__(self, kind: str):
        self.kind = kind

    def get_prompt(self, x: str, obj_str: str) -> str:
        """
        Given a SMLIES string, create a prompt string.

        Parameters:
        -----------
        x: str
            A string representation of x. For example, if we work with molecules, x could be the SMILES string.

        obj_str: str
            The textual name of the objective property. E.g. the "redox potential" or "solvation energy" of a molecule
        """
        raise NotImplementedError
